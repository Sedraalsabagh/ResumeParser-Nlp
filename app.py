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
def classify_links(links):
    classified = {
        'linkedin': [],
        'github': [],
        'gmail': [],
        'other': []
    }

    for link in links:
        lower_link = link.lower()
        if 'linkedin.com' in lower_link:
            classified['linkedin'].append(link)
        elif 'github.com' in lower_link:
            classified['github'].append(link)
        elif 'mailto:' in lower_link and 'gmail.com' in lower_link:
            classified['gmail'].append(link)
        else:
            classified['other'].append(link)

    return classified

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
def extract_links_from_pdf(pdf_path):
    links = []
    doc = fitz.open(pdf_path)
    for page in doc:
        for link in page.get_links():
            uri = link.get("uri", None)
            if uri:
                links.append(uri)
    return links
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

# --------------------------------Extract languge-------------------------------
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from langcodes import Language
def detect_language_name(text):
    try:
        lang_code = detect(text)
        return Language.get(lang_code).display_name()
    except LangDetectException:
        return "Unknown"

def extract_languages_with_levels_from_pdf(pdf_path):
    languages = []
    known_levels = ["native", "fluent", "advanced", "intermediate", "basic", "beginner", "b1", "b2", "c1", "c2", "a1", "a2"]
    known_languages = ["arabic", "english", "french", "german", "spanish"]

    full_text = ""

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if not text:
                continue
            full_text += "\n" + text  
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            in_languages_section = False
            
            for line in lines:
                if re.search(r'\blanguages\b', line, re.IGNORECASE):
                    in_languages_section = True
                    continue

                if in_languages_section:
                    if re.search(r'\b(skills|experience|projects|certifications|education)\b', line, re.IGNORECASE):
                        break

                    lower_line = line.lower()
                    for lang in known_languages:
                        if lang in lower_line:
                            level_found = None
                            for lvl in known_levels:
                                if lvl in lower_line:
                                    level_found = lvl.capitalize()
                                    break
                            languages.append({
                                "language": lang.capitalize(),
                                "level": level_found or "Unknown"
                            })

    # ✅ في حال لم يتم العثور على أي لغة من القسم المحدد
    if not languages:
        lower_full_text = full_text.lower()
        for lang in known_languages:
            if lang in lower_full_text:
                languages.append({
                    "language": lang.capitalize(),
                    "level": "Unknown"
                })

    return languages
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
def extract_experiences(text):
    experience_section_keywords = ['experience', 'work experience', 'professional experience']
    experiences = []

    lines = text.split('\n')
    experience_lines = []
    collecting = False

    for line in lines:
        line_clean = line.strip().lower()
        if any(keyword in line_clean for keyword in experience_section_keywords):
            collecting = True
            continue
        if collecting:
            if line.strip() == "" or line.lower().startswith("education") or line.lower().startswith("skills"):
                break  
            experience_lines.append(line.strip())

    current_exp = {}
    for line in experience_lines:
        date_match = re.search(r'(\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|'
                               r'May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|'
                               r'Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[\s\-.,]?\d{4})\s*(?:-|to)?\s*'
                               r'(\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|'
                               r'May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|'
                               r'Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[\s\-.,]?\d{4}|present)?', line, re.IGNORECASE)

        if date_match:
            if current_exp:
                experiences.append(current_exp)
                current_exp = {}
            current_exp['start_date'] = date_match.group(1)
            current_exp['end_date'] = date_match.group(2) if date_match.group(2) else "Present"
        elif 'at' in line.lower() and ' | ' not in line:
            parts = line.split(' at ')
            if len(parts) == 2:
                current_exp['job_title'] = parts[0].strip()
                current_exp['company'] = parts[1].strip()
        else:
            # وصف المهمة
            if 'description' in current_exp:
                current_exp['description'] += " " + line
            else:
                current_exp['description'] = line

    if current_exp:
        experiences.append(current_exp)

    return experiences

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
    experiences = extract_experiences(doc.text)
    language = extract_language(doc.text)  

    return {
        'name': name,
        'email': email,
        'phone': phone,
        'education': education,
        'skills': skills,
        'degree_major': degree_major,
        'experience': experience,
        'experiences': experiences,
        'language': language,  
    }

import pdfplumber
def extract_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    doc = nlp(text)

    # Basic information extraction
    name = extract_name(text)
    email = extract_email(doc)
    phone = extract_contact_number_from_resume(doc)
    education = extract_education_from_resume(doc)
    skills = extract_skills(doc)
    major = extract_major(doc)
    experience_info = extract_experience(doc)
    languages = extract_languages_with_levels_from_pdf(pdf_path)
    links = extract_links_from_pdf(pdf_path)
    classified_links = classify_links(links)

    # Compose final result dictionary
    result = {
        "name": name,
        "email": email,
        "phone": phone,
        "education": education,
        "skills": skills,
        "major": major,
        "experience_level": experience_info.get('level_of_experience'),
        "suggested_position": experience_info.get('suggested_position'),
        "languages": languages,
        "links": classified_links
    }

    return result

# ----------------------------------Example Usage-------------------------------
result = extract_from_pdf("data/Safa_Abou_zaid.pdf")
print(result)



import pdfplumber

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

pdf_path = "data/Safa_Abou_zaid.pdf"
text = extract_text_from_pdf(pdf_path)


print("="*40)
print("PDF TEXT CONTENT:")
print("="*40)
print(text)
print("="*40)