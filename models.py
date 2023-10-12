from typing import List, Dict
from pydantic import BaseModel, Field

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