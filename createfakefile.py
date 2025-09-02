from faker import Faker
from faker.providers import BaseProvider
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
from langchain_ollama.llms import OllamaLLM

load_dotenv()

llm = OllamaLLM(model="qwen3:8b")
class ChineseProductProvider(BaseProvider):
    def product_name(self):
        products = ["智能手机", "笔记本电脑", "智能手表", "无线耳机", "平板电脑"]
        return self.random_element(products)

fake = Faker('zh_CN')
fake.add_provider(ChineseProductProvider)



if __name__ == "__main__":
    name = fake.name()
    reporter = fake.name()
    product = fake.product_name()
    company = fake.company()
    job = fake.job()
    prompt_template = f"""
    你是一名专业的新闻撰稿人，根据给出的人名{name}，公司{company}，岗位{job}，产品{product}， 记者{reporter}，请你写一段新闻。
    新闻需要拥有3-5个分块，全文长度应该在2000字以上，每一块内容应该在500字以上，其中以上元素之间的关系可以自由发挥，例如人不一定是产品的生产者，也可以是使用者，销售者等等等等。
    每一块分块之间的内容需要做出差异化。
    新闻需要符合中文的语法规范，例如使用正确的标点符号、使用中文的语法结构等，新闻内容应该遵循markdown格式。
    新闻总标题应该写成`# 新闻标题` 的样式，例如`# 新闻标题`，`# 新闻内容`，`# 新闻结尾`等。
    新闻总标题应当是对新闻内容的概括。
    每个分块的标题应该写成`## 分块标题`的样式，例如`## 新闻标题`，`## 新闻内容`，`## 新闻结尾`等。
    分块标题应当是对分块内容的概括。
    注意：你只需要输出新闻，不需要输出别的部分，如思考过程等。
    新闻格式范例如下：
    # 标题
    ## 分块一
    内容一...
    ## 分块二
    内容二...
    ## 分块三
    内容三...
    """

    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | llm | StrOutputParser()
    news = chain.invoke({"name": name, "company": company, "job": job, "product": product})
    print(news)

