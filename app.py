import re
import fitz  # PyMuPDF
import spacy
import csv
import nltk
from spacy.matcher import Matcher

nltk.download('punkt')

# Load spaCy English model
nlp = spacy.load('en_core_web_sm')

def load_keywords(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        return set(row[0] for row in reader)

# ----------------------------------Extract Name----------------------------------
def extract_name(text):
    nlp_text = nlp(text)
    matcher = Matcher(nlp.vocab)
    pattern = [{'POS': 'PROPN'}, {'POS': 'PROPN'}]
    matcher.add('NAME', [pattern])

    matches = matcher(nlp_text)
    for match_id, start, end in matches:
        span = nlp_text[start:end]
        return span.text
    return None
# --------------------------------------------------------------------------------

# ----------------------------------Extract Email---------------------------------
def extract_email(doc):
    matcher = spacy.matcher.Matcher(nlp.vocab)
    email_pattern = [{'LIKE_EMAIL': True}]
    matcher.add('EMAIL', [email_pattern])

    matches = matcher(doc)
    for match_id, start, end in matches:
        if match_id == nlp.vocab.strings['EMAIL']:
            return doc[start:end].text
    return ""
# --------------------------------------------------------------------------------

# ----------------------------------Extract Phone Number--------------------------
def extract_contact_number_from_resume(doc):
    text = doc.text
    pattern = r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
    match = re.search(pattern, text)
    if match:
        return match.group()
    return None
# --------------------------------------------------------------------------------

# --------------------------------Extract Education-------------------------------
def extract_education_from_resume(doc):
    universities = []
    for entity in doc.ents:
        if entity.label_ == "ORG" and ("university" in entity.text.lower() or "college" in entity.text.lower() or "institute" in entity.text.lower()):
            universities.append(entity.text)
    return universities
# --------------------------------------------------------------------------------

# ----------------------------------Extract Skills--------------------------------
def csv_skills(doc):
    skills_keywords = load_keywords('data/newSkills.csv')
    skills = set()

    for keyword in skills_keywords:
        if keyword.lower() in doc.text.lower():
            skills.add(keyword)

    return skills

nlp_skills = spacy.load('TrainedModel/skills')

def extract_skills_from_ner(doc):
    non_skill_labels = {'DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL', 'EMAIL'}
    skills = set()
    for ent in nlp_skills(doc.text).ents:
        if ent.label_ == 'SKILL' and ent.label_ not in non_skill_labels and not ent.text.isdigit():
            skill_text = ''.join(filter(str.isalpha, ent.text))
            if skill_text:
                skills.add(skill_text)
    return skills

def is_valid_skill(skill_text):
    return len(skill_text) > 1 and not any(char.isdigit() for char in skill_text)

def extract_skills(doc):
    skills_csv = csv_skills(doc)
    skills_ner = extract_skills_from_ner(doc)  # comment out if no NER model
    filtered_skills_ner = {skill for skill in skills_ner if is_valid_skill(skill)}
    filtered_skills_csv = {skill for skill in skills_csv if is_valid_skill(skill)}

    combined_skills = filtered_skills_csv.union(filtered_skills_ner)
    combined_skills = filtered_skills_csv

    return list(combined_skills)
# --------------------------------------------------------------------------------

# ----------------------------------Extract Major---------------------------------
def extract_major(doc):
    major_keywords = load_keywords('data/majors.csv')
    for keyword in major_keywords:
        if keyword.lower() in doc.text.lower():
            return keyword
    return ""
# --------------------------------------------------------------------------------

# --------------------------------Extract Experience-------------------------------
def extract_experience(doc):
    verbs = [token.text.lower() for token in doc if token.pos_ == 'VERB']

    senior_keywords = ['lead', 'manage', 'direct', 'oversee', 'supervise', 'orchestrate', 'govern']
    mid_senior_keywords = ['develop', 'design', 'analyze', 'implement', 'coordinate', 'execute', 'strategize']
    mid_junior_keywords = ['assist', 'support', 'collaborate', 'participate', 'aid', 'facilitate', 'contribute']

    if any(keyword in verbs for keyword in senior_keywords):
        level_of_experience = "Senior"
    elif any(keyword in verbs for keyword in mid_senior_keywords):
        level_of_experience = "Mid-Senior"
    elif any(keyword in verbs for keyword in mid_junior_keywords):
        level_of_experience = "Mid-Junior"
    else:
        level_of_experience = "Entry Level"

    suggested_position = suggest_position(verbs)

    return {
        'level_of_experience': level_of_experience,
        'suggested_position': suggested_position
    }
# --------------------------------------------------------------------------------

# -----------------------------------Suggestions----------------------------------
def load_positions_keywords(file_path):
    positions_keywords = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            position = row['position']
            keywords = [keyword.lower() for keyword in row['keywords'].split(',')]
            positions_keywords[position] = keywords
    return positions_keywords

def suggest_position(verbs):
    positions_keywords = load_positions_keywords('data/position.csv')
    for position, keywords in positions_keywords.items():
        if any(keyword in verbs for keyword in keywords):
            return position
    return "Position Not Identified"

def extract_resume_info_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page_num in range(doc.page_count):
        page = doc[page_num]
        text += page.get_text()
    return nlp(text)

def extract_resume_info(doc):
    name = extract_name(doc)
    email = extract_email(doc)
    phone = extract_contact_number_from_resume(doc)
    education = extract_education_from_resume(doc)
    skills = extract_skills(doc)
    degree_major = extract_major(doc)
    experience = extract_experience(doc)

    return {
        'name': name,
        
        'email': email,
        'phone': phone,
        'education': education,
        'skills': skills,
        'degree_major': degree_major,
        'experience': experience
    }

def main():
    # ضع هنا اسم ملف السيرة الذاتية الموجود في مجلد data
    resume_path = 'data/Safa_Abou_zaid.pdf'  # غيره إلى اسم ملف لديك

    print(f"تحميل السيرة الذاتية من: {resume_path}")
    doc = extract_resume_info_from_pdf(resume_path)
    info = extract_resume_info(doc)

    print(f"Full Name: {info['name'] if info['name'] else 'Not Found'}")
    print(f"Email: {info['email'] if info['email'] else 'Not Found'}")
    print(f"Phone Number: {info['phone'] if info['phone'] else 'Not Found'}")
    print(f"Education (Universities/Institutes): {', '.join(info['education']) if info['education'] else 'Not Found'}")
    print(f"Skills: {', '.join(info['skills']) if info['skills'] else 'Not Found'}")
    print(f"Experience Level: {info['experience']['level_of_experience']}")
    print(f"Suggested Position: {info['experience']['suggested_position']}")

if __name__ == '__main__':
    main()
