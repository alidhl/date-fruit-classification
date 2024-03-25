from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
class FactGenerator:
    def __init__(self):
        chat = ChatOpenAI(streaming=True)
        prompt_template = "You are world class expert of date palm fruits. Generate a intersting fact about the following date fruit type {label}"
        prompt = ChatPromptTemplate.from_template(prompt_template)
        self.chain = {"label": RunnablePassthrough()} | prompt | chat | StrOutputParser()
        
    def get_chain(self):
        return self.chain
    
    
# Testing the FactGenerator
if __name__ == '__main__':
    fact_generator = FactGenerator()
    chain = fact_generator.get_chain()
    output = chain.invoke("NabtatAli")
    print(output)  # Expected output: 'Fact about NabtatAli'