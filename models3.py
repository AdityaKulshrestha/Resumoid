from langchain.output_parsers import StructuredOutputParser, ResponseSchema

resume_response_schemas = [
    ResponseSchema(name="Education", field_type=list, description="List of Education Degress earned by the person in resume"),
    ResponseSchema(name="Experience", description="List of different companies and tasks the person has worked for."),
    ResponseSchema(name="Personal Details", description="Personal details of the candidate in the resume"),
    ResponseSchema(name="skills", description="Identify and extract a list of skills mentioned in the resume"),
    ResponseSchema(name="projects", description="Projects and their description that the candidate has built")

]
resume_output_parser = StructuredOutputParser.from_response_schemas(resume_response_schemas)