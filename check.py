import streamlit as st
from typing import List, Dict
import re
import PyPDF2
import base64
import matplotlib.pyplot as plt
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from pydantic import BaseModel, Field
from langchain.chains.conversation.memory import ConversationBufferMemory
from dotenv import load_dotenv

load_dotenv()

from models2 import *

# Defining LLM
llm = ChatOpenAI(model = "gpt-3.5-turbo")

def read_pdf(file):
    """
    Reads a resume in PDF file and extract text from it.
    :param file: File object
    :return: String
    """
    reader = PyPDF2.PdfReader(file)
    num_pages = len(reader.pages)
    text = ""
    for i in range(num_pages):
        page = reader.pages[i]
        text += page.extract_text()
    return text

# def create_chart_overall(value: int):
#     """
#     Return matplotlib.pyplot figure.
#     :param value: Intger value.
#     :return:
#     """
#     fig, ax = plt.subplots(3,3)

#     value = value*10

#     sizes = [10 - value, value]

#     # Create a pie chart
#     ax.pie(sizes, colors=['red', 'blue'], startangle=90)

#     # Draw a white circle in the center
#     centre_circle = plt.Circle((0, 0), 0.70, fc='white')
#     ax.add_artist(centre_circle)
#     ax.text(0, 0, f'{value}/100', horizontalalignment='center', verticalalignment='center')

#     return fig

def create_chart_overall(value: int):
    """
    Return matplotlib.pyplot figure.
    :param value: Integer value.
    :return:
    """
    fig, ax = plt.subplots(figsize=(3, 3))

    value = value*10
    sizes = [100 - value, value]

    # Define colors (blue for the score, silver for the remaining)
    colors = ['silver', 'blue']

    # Define explode parameters to separate the score section slightly
    explode = (0, 0.1)

    # Create a pie chart with shadows for a 3D effect
    ax.pie(sizes, explode=explode, colors=colors, startangle=90, shadow=True)

    # Draw a white circle in the center
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    ax.add_artist(centre_circle)
    ax.text(0, 0, f'{value}/100', horizontalalignment='center', verticalalignment='center')

    # Equal aspect ratio ensures that pie is drawn as a circle
    ax.axis('equal')

    return fig


def create_chart(value: int):
    """
    Return matplotlib.pyplot figure.
    :param value: Intger value.
    :return:
    """
    fig, ax = plt.subplots()

    sizes = [10 - value, value]

    # Create a pie chart
    ax.pie(sizes, colors=['red', 'green'], startangle=90)

    # Draw a white circle in the center
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    ax.add_artist(centre_circle)
    ax.text(0, 0, f'{value}/10', horizontalalignment='center', verticalalignment='center')

    return fig

def extract_info(resume: str):
    """
    Extracts sections from the resume
    :param resume:
    :return:
    """
    parser = OutputFixingParser.from_llm(parser=PydanticOutputParser(pydantic_object=Resume), llm=llm)
    format_instructions = parser.get_format_instructions()
    resume_text = llm.predict(
        f"Given a resume {resume} \n Extract all the relevant sections.  \n {format_instructions}")
    resume_info = parser.parse(resume_text)
    return resume_info


# resume_text = read_pdf("sample_cv.pdf")
# resume_info = extract_info(resume_text)

# print(resume_info)
# print(type(resume_info))
# print(resume_info.personal_details)
# print(resume_info.personal_details.name)
# print(resume_info.personal_details.email)
# print(resume_info.education)
# print(type(resume_info.education))
# print(resume_info.experience)
# print(type(resume_info.experience))

def description_evaluation(resume, job_description):
    prompt_template = f'''You are an Resume Expert. Your job is to give feedback on the resume based on the provided job description.
    Be specific about the points.
    
    Resume: {resume}
    
    Job Description: {job_description}
    
    Please provide the feedback in the following format.
    
    ## Strengths:
    <list strengths here>
    
    ## Weaknesses:
    <list weaknesses here>
    
    ## Recommendations to improve CV:
    <list recommendations here>
    
    
    
    ONLY QUOTE THE INFORMATION PROVIDED IN THE RESUME. DO NOT MAKE UP INFORMATION WHICH IS NOT EXPLICITLY PROVIDED IN RESUME.
    RETURN THE RESPONSE IN MARKDOWN FORMAT IN BULLET POINTS.
    '''
    output = llm.predict(prompt_template)
    return output


def llm_scoring(llm, resume_text, job_description):
    # Define the prompt
    prompt = f"""
    Given the following resume for the job role '{job_description}', please evaluate and provide a score between 1 to 10 (where 1 is the lowest and 10 is the highest), and provide feedback for each category and the overall resume:

    {resume_text}

    Categories:
    1. Relevant Experience
    2. Education
    3. Skills
    4. Projects

    Please provide the scores and feedback to the candidate in the following format:

    Relevant Experience: {{score_experience}}, Feedback: {{feedback_experience}}
    Education: {{score_education}}, Feedback: {{feedback_education}}
    Skills: {{score_skills}}, Feedback: {{feedback_skills}}
    Projects: {{score_projects}}, Feedback: {{feedback_projects}}
    Overall Score: {{score_overall}}, Feedback: {{feedback_overall}}
    """
    # Ask the LLM to score the resume and provide feedback
    response = llm.predict(prompt)

    parser = OutputFixingParser.from_llm(parser=PydanticOutputParser(pydantic_object=ResumeScores), llm=llm)
    format_instructions = parser.get_format_instructions()

    resume_scores = parser.parse(response)

    return resume_scores


def main():
    st.set_page_config(layout="wide")
    st.title("Welcome to Resumoid 🤖")
    st.header("Your personal AI ATS!")

    st.markdown("📄 Upload your resume and job role to get feedback")

    resume_pdf = st.file_uploader("Upload your resume", type=['pdf'], label_visibility='collapsed')
    job_description = st.text_input("Enter the role for which you are applying")

    submit = st.button("Submit")

    if resume_pdf and job_description and submit:
        resume_text = read_pdf(resume_pdf)
        resume_info = extract_info(resume_text)
        gpt4_model = ChatOpenAI(model = 'gpt-4')
        resume_scores = llm_scoring(llm = gpt4_model, resume_text = resume_text, job_description= job_description)

        st.divider()

        st.markdown("### Candidate Details")

        # st.text(resume_info)
        # st.write(resume_info)
        # print(resume_info)
        # name, phone, email

        # name = resume_info.personal_details.name
        # st.markdown("### Candidate Details")
        st.markdown("**Name:** " + resume_info.personal_details.name)
        # st.markdown("**Name:** " + name)
        st.markdown("**Email:** " + resume_info.personal_details.email)
        st.markdown("**Contact Number:** " + resume_info.personal_details.contact_num)
        st.markdown("**University:** " + resume_info.education[0].university)
        st.markdown("**Current Job Role:** " + resume_info.experience[0].company_name)
        st.markdown("**Company:** " + resume_info.experience[0].job_role)

        st.divider()

        ocol1, ocol2, ocol3 = st.columns(3)

        ocol2.markdown("### Relevance Score \n\n\n\n")
        ocol2.pyplot(create_chart_overall(resume_scores.overall_score))
        ocol2.markdown(resume_scores.overall_feedback)

        st.divider()

        st.markdown("### Evaluation")

        st.text(f"Here is the evaluation of your resume for the {job_description} role.")

        

        # st.markdown("**Experience Score:** " + str(resume_scores.experience_score))
        # st.markdown("**Experience Feedback:** " + resume_scores.experience_feedback)
        # st.markdown("**Education Score:** " + str(resume_scores.education_score))
        # st.markdown("**Education Feedback:** " + resume_scores.education_feedback)

        col1, col2, col3, col4 = st.columns(4)
        # Column 1
        col1.markdown("### Experience \n\n\n")
        col1.pyplot(create_chart(resume_scores.experience_score))
        col1.markdown(resume_scores.experience_feedback)

        # Column 2
        col2.markdown("### Education \n\n\n")
        col2.pyplot(create_chart(resume_scores.education_score))
        col2.markdown(resume_scores.education_feedback)

        # Column 3
        col3.markdown("### Skills \n\n\n\n")
        col3.pyplot(create_chart(resume_scores.skills_score))
        col3.markdown(resume_scores.skills_feedback)

        # Column 4
        col4.markdown("### Projects \n\n\n\n")
        col4.pyplot(create_chart(resume_scores.projects_score))
        col4.markdown(resume_scores.projects_feedback)


        st.divider()

        st.markdown("### Detailed Comments")
        feedback_jobdesc = description_evaluation(resume_text, job_description)
        st.markdown(feedback_jobdesc)

        if "expert_chat" not in st.session_state:
            st.session_state.expert_chat = False

        if st.button("Ask an Expert"):
                st.session_state.expert_chat = True

        if st.session_state.expert_chat:
            query = st.text_input("Enter your query", placeholder= "enter your query", label_visibility= "collapsed")
            if query:
                # st.write(query)
                st.write("Expert Chat coming soon!")














if __name__ == '__main__':
    main()

