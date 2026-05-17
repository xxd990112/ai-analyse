import os
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel as RequestBaseModel, BaseModel, Field
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

# --------------------------
# 火山方舟配置（已为你填好）
# --------------------------
API_KEY = "sk-c7pq5t12dp7cxu4sasfv7evumf3b3aqi3bn1njao9hevqmjv"
BASE_URL = "https://api.xiaomimimo.com/v1"
MODEL_NAME = "mimo-v2-flash"

# 初始化FastAPI
app = FastAPI(title="AI 商品智能对比工具 - 中文版后端")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化LLM（豆包模型）
llm = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model=MODEL_NAME,
    temperature=0.7
)

# 数据模型定义
class ProductInfo(BaseModel):
    id: str = Field(description="随机ID")
    name: str = Field(description="商品全称")
    brand: str = Field(description="品牌")
    price: str = Field(description="带¥的价格字符串")
    priceNum: float = Field(description="纯数字价格")
    rating: float = Field(description="评分0-5")
    features: int = Field(description="功能得分1-10")
    reputation: float = Field(description="口碑得分1-10")
    url: str = Field(description="购买链接")
    release_date: str = Field(description="发布时间或时期，如 '2023-09' 或 '经典款'")
    specs: Dict[str, str] = Field(description="商品详细参数，例如 {'处理器': 'A17 Pro', '内存': '8GB'}")

class RecommendationCategory(BaseModel):
    category: str = Field(description="类别名称，如 '智能手机'")
    products: List[ProductInfo] = Field(description="该类别下的推荐产品列表")

class RecommendationResult(BaseModel):
    recommendations: List[RecommendationCategory] = Field(description="不同类别的推荐列表")

class SearchResult(BaseModel):
    products: List[ProductInfo] = Field(description="搜索到的产品列表")

class ProductAnalysis(BaseModel):
    recommended_product: str = Field(description="推荐的商品名称")
    comparison_summary: str = Field(description="对比总结")
    advantages: List[str] = Field(description="核心优势")
    disadvantages: List[str] = Field(description="不足之处")
    radar_data: Dict[str, List[int]] = Field(description="雷达图数据，格式：{产品名：[价格, 性能, 功能, 外观, 续航, 口碑]}，得分均为0-100")
    detailed_comparison: Dict[str, Dict[str, str]] = Field(description="全方位横向对比数据，格式：{维度名称: {产品A名: '参数值', 产品B名: '参数值'}}，维度应包含处理器、屏幕、电池、影像等关键参数")

class SearchRequest(RequestBaseModel):
    query: Optional[str] = None
    products: Optional[List[Dict[str, Any]]] = None

# 解析器
search_parser = PydanticOutputParser(pydantic_object=SearchResult)
compare_parser = PydanticOutputParser(pydantic_object=ProductAnalysis)
recommend_parser = PydanticOutputParser(pydantic_object=RecommendationResult)

# 接口定义
@app.get("/api/recommendations")
async def get_recommendations():
    try:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个全能的商品采购专家。请推荐3个热门品类（如：智能手机、办公笔记本、真无线耳机），每个品类推荐3-4款当前中国市场最热门的产品，并提供详细参数。\n\n{format_instructions}"),
            ("human", "请提供热门商品推荐。")
        ]).partial(format_instructions=recommend_parser.get_format_instructions())

        chain = prompt | llm | recommend_parser
        result = chain.invoke({})
        return {
            "status": "success",
            "data": result.model_dump()["recommendations"]
        }
    except Exception as e:
        print("推荐接口错误:", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/search")
async def analyze_products(request: SearchRequest):
    try:
        print("收到的请求数据:", request.model_dump())

        # 情况1：用户发送的是产品列表，进行对比分析
        if request.products:
            prompt = ChatPromptTemplate.from_messages([
                ("system", "你是一个专业的商品对比专家。请分析以下商品并给出深度对比和雷达图得分（0-100）。\n\n{format_instructions}"),
                ("human", "请分析以下商品：{products}")
            ]).partial(format_instructions=compare_parser.get_format_instructions())

            chain = prompt | llm | compare_parser
            result = chain.invoke({"products": request.products})
            return {
                "status": "success",
                "data": result.model_dump()
            }

        # 情况2：用户发送的是关键词，进行产品搜索
        elif request.query:
            # 仅基于关键词生成结构化的候选产品清单（不包含网页抓取逻辑）
            prompt = ChatPromptTemplate.from_messages([
                ("system", "你是一个全能的商品采购专家和资深数码分析师。请根据用户的搜索关键词，生成 5-8 款具有代表性的候选商品清单，用于后续对比分析。要求：\n1. **全量整理**：输出 5-8 款具有代表性的商品。\n2. **时效多样性**：尽量覆盖新品、经典款与仍具性价比的老款（如果适用）。\n3. **档次覆盖**：包含高端旗舰、主流中端和入门级产品，形成全面矩阵。\n4. **结构化输出**：补全 specs（CPU/屏幕/电池/影像/内存等核心参数），给出 release_date（可为“经典款/近一年”等），并给出合理评分。\n5. **格式规范**：价格必须包含‘¥’符号，ID 必须随机唯一。\n\n{format_instructions}"),
                ("human", "搜索关键词：{query}")
            ]).partial(format_instructions=search_parser.get_format_instructions())

            chain = prompt | llm | search_parser
            result = chain.invoke({"query": request.query})
            return {
                "status": "success",
                "data": result.model_dump()["products"]
            }

        else:
            return {"status": "error", "message": "请求参数不足"}

    except Exception as e:
        print("接口错误:", str(e))
        raise HTTPException(status_code=500, detail=str(e))

# 启动服务
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
