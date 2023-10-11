import streamlit as st
from typing import List, Dict
import PyPDF2
import base64
import matplotlib.pyplot as plt
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain.chains.conversation.memory import ConversationBufferMemory
from dotenv import load_dotenv

load_dotenv()


class PersonalDetails(BaseModel):
    name: str = Field(description="Name of the person")
    email: str = Field(description="Email id of the person")
    contact_num: str = Field(description="Phone Number of the person")


class Sections(BaseModel):
    personal_details: PersonalDetails = Field(description="Personal Details ")
    education: str = Field(description="Education section")
    experience: str = Field(description="Experience section")


class Education(BaseModel):
    text: str = Field(description="Education section and details")


class EducationList(BaseModel):
    educationList: List[Education] = Field(description="List of education")


class Projects(BaseModel):
    projectname: str = Field(description="Title of the project")
    description: str = Field(description="Description of the project")


class ProjectsList(BaseModel):
    projects: List[Projects] = Field(description="List of projects")


class accTasks(BaseModel):
    workDone: str = Field(description="Work accomplished or tasks done at the job.")


class Experience(BaseModel):
    companyName: str = Field(description="Name of the company")
    duration: str = Field(description="Duration of the internship/job.")
    jobRole: str = Field(description="Name of the role.")
    tasksList: List[accTasks] = Field(description="Tasks or work done at the company.")


class ExperiencesList(BaseModel):
    experiences: List[Experience] = Field(description="List of the experiences")


class Recommendation(BaseModel):
    score: int = Field(description="Score given under the criteria")
    suggestion: str = Field(description="Suggestion for improvement on the given criteria.")


class RecommendationList(BaseModel):
    recommendationsList: List[Recommendation] = Field(description="List of recommendation given under each criteria")


class ExperienceEvaluation(BaseModel):
    experience: List[Experience] = Field(description="Details about the job/intern at a company")
    suggestionList: List[Recommendation] = Field(description="List of recommendation for the given experience.")


# Defining LLM
llm = ChatOpenAI()


def extract_section(resume: str):
    """
    Extracts sections from the resume
    :param resume:
    :return:
    """
    parser = PydanticOutputParser(pydantic_object=Sections)
    format_instructions = parser.get_format_instructions()
    section_text = llm.predict(
        f"Given a resume {resume} \n Extract the section from the resume. \n {format_instructions}")
    resume_sections = parser.parse(section_text)
    return resume_sections


def analyse_education_section(text: str):
    """
    Analyze the resume context
    :param text: Resume Text
    :return:
    """
    parser = PydanticOutputParser(pydantic_object=EducationList)
    format_instructions = parser.get_format_instructions()
    output_education_section = llm.predict(
        f"Given an education section from a resume: {text}. \n Extract the degrees and colleges from where the user "
        f"has acquired his/her education. \n {format_instructions}")
    listofEducation = parser.parse(output_education_section)
    return listofEducation


def analyse_experience_section(experience: str):
    """
    Analyses and recommends suggestions for experience section.
    :param experience:
    :return:
    """
    parser = PydanticOutputParser(pydantic_object=RecommendationList)
    format_instructions = parser.get_format_instructions()
    prompt_template = f"""Given is a experience profile of a person. 
    {experience} Based on the criteria below. First score the resume for each given criteria out of 10 and then suggest 
    improvements to the resume based on the criteria. 
    Important Instruction: 
    1. Be specific about teach suggestion. 
    2. Suggestion should be according to the given experience. 
    
    Criteria 1 - Quantification evidence of impact 
 
    Criteria 2 - Less repetition and Unique Action Word.
    
    Criteria 3 - Weak Action Verbs
    
    Criteria 4 - Avoided Responsibility-oriented Words
 
    Criteria 5 - Buzzwords & Clichés
 
    \n
    {format_instructions}
    """

    evaluated_output = llm.predict(prompt_template)
    structured_output = parser.parse(evaluated_output)
    return structured_output


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


def displayPDF(file):
    """
    Function to display the PDF
    :param file: File object
    :return: None
    """
    # Reading file
    bytes_data = file.getvalue()
    # To convert bytes to base64
    base64_pdf = base64.b64encode(bytes_data).decode()

    # Embedding PDF in HTML
    pdf_display = F"""<div style="display: flex; justify-content: center; align-items: center;">
    <iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>
</div>
"""
    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)


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


def description_evaluation(resume, job_description):
    prompt_template = f'''You are an Resume Expert. Your job is to give feedback on the resume based on the provided job description.
    Be specific about the points.
    
    Resume: {resume}
    
    Job Description: {job_description}
    
    Please provide the feedback in the following format
    
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


def main():
    st.set_page_config(layout="wide")
    st.title("Welcome to Resumoid 🤖")
    st.header("Your personal AI ATS!")

    st.markdown("📄 Upload your resume and job role to get feedback")

    resume_pdf = st.file_uploader("Upload your resume", type=['pdf'], label_visibility='collapsed')
    job_description = st.text_input("Enter the role for which you are applying")

    submit = st.button("Submit")

    if resume_pdf and job_description and submit:
        displayPDF(resume_pdf)
        st.divider()
        resume = read_pdf(resume_pdf)
        resume_sections = extract_section(resume)
        details = resume_sections.personal_details
        education = resume_sections.education
        experience = resume_sections.experience
        experience_evaluation = analyse_experience_section(experience)
        st.markdown(f"""
        ### Name - {details.name}
        ### Phone Number - {details.contact_num}
        ### Email Address - {details.email}
        """)
        section1, section2 = st.columns(2)
        section1.markdown(f"""# Education \n{education}""")
        section2.markdown(f"""# Experience \n{experience}""")

        st.divider()
        st.markdown("## Detailed Analysis")

        col1, col2, col3, col4, col5 = st.columns(5)
        # Column 1
        col1.markdown("### Quantification evidence of impace\n\n\n")
        col1.pyplot(create_chart(experience_evaluation.recommendationsList[0].score))
        col1.markdown(experience_evaluation.recommendationsList[0].suggestion)

        # Column 2
        col2.markdown("### Less repetition and Unique Action Word.\n\n\n")
        col2.pyplot(create_chart(experience_evaluation.recommendationsList[1].score))
        col2.markdown(experience_evaluation.recommendationsList[1].suggestion)

        # Column 3
        col3.markdown("### Weak Action Verbs\n\n\n\n")
        col3.pyplot(create_chart(experience_evaluation.recommendationsList[2].score))
        col3.markdown(experience_evaluation.recommendationsList[2].suggestion)

        # Column 4
        col4.markdown("### Avoided Responsibility-oriented Words\n\n\n\n")
        col4.pyplot(create_chart(experience_evaluation.recommendationsList[3].score))
        col4.markdown(experience_evaluation.recommendationsList[3].suggestion)

        # Column 5
        col5.markdown("### Buzzwords & Clichés\n\n\n\n\n")
        col5.pyplot(create_chart(experience_evaluation.recommendationsList[4].score))
        col5.markdown(experience_evaluation.recommendationsList[4].suggestion)

        st.divider()
        st.markdown("## Feedback on the resume based on job description!")
        feedback_jobdesc = description_evaluation(resume, job_description)
        st.markdown(feedback_jobdesc)

    with st.sidebar:
        if resume_pdf and job_description:
            st.button("Chat with an expert!")


if __name__ == '__main__':
    main()
