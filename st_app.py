from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
import os
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
import streamlit as st
load_dotenv()



os.environ["AZURE_OPENAI_API_KEY"] =os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")



model = AzureChatOpenAI(
    azure_deployment="gpt-4o-mini",  # Deployment name from Azure
    azure_endpoint=endpoint,
    api_version="2024-05-01-preview",
    temperature=0.7
)


def generate_story(location, name,language):
    

    template_story = """
    As a childern's book writer, please come up with a simple and fun story for kids.
    lullaby based on the location
    {location}
    and the main character {name}


    STROY:
    """


    prompt = PromptTemplate(input_variables=["location", "name"],
                            template=template_story)

    chain_story =LLMChain(llm=model, 
                          prompt=prompt,
                          output_key="story")


    template2 = """ translate this {story} into {langu}

    """

    prompt_trans = PromptTemplate(input_variables=["story","langu"], template=template2)

    chain_trans = LLMChain(llm=model , prompt=prompt_trans , output_key="translated_story")

    overall_chain = SequentialChain(chains=[chain_story,chain_trans],
                                input_variables=["location","name", "langu"],
                                output_variables=["story","translated_story"],
                                verbose=True)


    res =overall_chain({"location":location,"name":name,"langu":language})


    return res


def main():
    st.set_page_config(page_title="Children's Story Generator",layout="centered")
    st.title("Children's Story by AI 👶")

    loc_input = st.text_input("Enter the location for the story:")
    name_input = st.text_input("Enter the name of the main character:")
    langu_input = st.text_input("Enter the language to translate the story into:")


    if st.button("Generate Story"):
        if loc_input and name_input and langu_input:
            with st.spinner("Generating story..."):
                result = generate_story(loc_input, name_input, langu_input)
                st.success("Story generated successfully!")
                st.subheader("Generated Story:")
                st.write(result["story"])
                st.subheader(f"Translated Story in {langu_input}:")
                st.write(result["translated_story"])
        else:
            st.error("Please fill in all fields.") 




    pass


if __name__ == "__main__":
    main()
