from typing import List, Optional
from pydantic import BaseModel, Field
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

resume_response_schemas = [
    ResponseSchema(name="Education", field_type=list,
                   description="List of Education Degress earned by the person in resume"),
    ResponseSchema(name="Experience", description="List of different companies and tasks the person has worked for."),
    ResponseSchema(name="Personal Details", description="Personal details of the candidate in the resume"),
    ResponseSchema(name="skills", description="Identify and extract a list of skills mentioned in the resume"),
    ResponseSchema(name="projects", description="Projects and their description that the candidate has built")

]
resume_output_parser = StructuredOutputParser.from_response_schemas(resume_response_schemas)


class PersonalDetails(BaseModel):
    name: str = Field(description="Name of the person")
    email: str = Field(description="Email id of the person")
    contact_num: str = Field(description="Phone Number of the person")


class Education(BaseModel):
    university: str = Field(description="Name of the university")
    degree: str = Field(description="Degree obtained")
    year_of_passing: Optional[str] = Field(description="Year of passing")
    field_of_study: Optional[str] = Field(description="Field of study")
    grade: Optional[str] = Field(description="Grade obtained")


class Project(BaseModel):
    project_name: str = Field(description="Title of the project")
    description: str = Field(description="Description of the project")


class Skill(BaseModel):
    skill_name: str = Field(description="Name of the skill")
    proficiency_level: Optional[str] = Field(description="Proficiency level of the skill")


class WorkTask(BaseModel):
    task: str = Field(description="Task performed at the job")


class Experience(BaseModel):
    company_name: str = Field(description="Name of the company")
    job_role: str = Field(description="Job role at the company")
    duration: str = Field(description="Duration of the job")
    tasks: List[WorkTask] = Field(description="List of tasks performed at the job")


class Resume(BaseModel):
    personal_details: PersonalDetails
    education: List[Education]
    experience: List[Experience]
    skills: List[Skill]
    projects: List[Project]


class ResumeScores(BaseModel):
    experience_score: int = Field(..., ge=1, le=10)
    experience_feedback: str
    education_score: int = Field(..., ge=1, le=10)
    education_feedback: str
    overall_score: int = Field(..., ge=1, le=10)
    overall_feedback: str


class Suggestion(BaseModel):
    original_task: List = Field(description="List of original work task mentioned in experience.")
    reframed: List = Field(description="List of corresponding reframed work task")


class SuggestionList(BaseModel):
    suggestionList: List[Suggestion]


class Resume_API(BaseModel):
    resume_eval: ResumeScores
    suggestions: SuggestionList
