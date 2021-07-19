# Using StanfordNER library

import re
import pprint
from datetime import datetime
from enum import Enum
import dateparser
import stanza

stanza.download('en')
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords

SECONDS_IN_YEAR = 31536000
nlp = stanza.Pipeline('en')


def clean_text(text):
    review = re.sub('(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?"', ' ', text)
    # review = review.lower()
    # review = review.split()
    # lm = WordNetLemmatizer()
    # review = [lm.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    # review = ' '.join(review)

    return review


def avg(ls):
    return sum(ls) / len(ls)


# Return keyword that is closer to most of the samples without the selected keyword neglecting samples
def weighted_keyword(keyword_pos, sample_pos) -> int:
    res = []
    for i in range(len(keyword_pos)):
        res.append(0)
        for j in range(len(sample_pos)):
            dist = keyword_pos[i] - sample_pos[j]
            res[i] += -dist * dist if dist < 0 else dist  # If keyword is beyond sample then punish
    return keyword_pos[res.index(min(res))]


# zipped_interval_overlap([1, 4, 7], [2, 3, 5]) -> {1: [2, 3], 4: [5], 7: []}
def zipped_interval_overlap(list_1, list_2) -> dict:
    res = {key: [] for key in list_1}
    interval_list = []
    for i in range(len(list_1) - 1):
        interval_list.append(list_1[i + 1] - list_1[i])
    max_len = max(interval_list)  # Use to include values from list_2 for the last index of list_1

    for i in range(len(list_1)):
        for j in range(len(list_2)):
            if i == len(list_1) - 1:
                if list_1[i] < list_2[j] <= list_1[i] + max_len:
                    res[list_1[i]].append(list_2[j])
            elif list_1[i] < list_2[j] < list_1[i + 1]:
                res[list_1[i]].append(list_2[j])
    return res


def min_interval(ls) -> int:
    res = []
    for i in range(len(ls) - 1):
        res.append(ls[i + 1] - ls[i])

    # return avg([avg(res), min(res)])
    return avg(res)


# Returns negative if n is greater than most, positive if n is less than most
def majority_partition(n: int, arr) -> int:
    res = 0
    for i in arr:
        if n < i:
            res += 1
        elif n > i:
            res -= 1
    return res


class Filled(Enum):
    MONTH = 1
    YEAR = 2
    NONE = 3
    BOTH = 4


# Returns the positions not filled in dates
def date_fill(date: datetime, today: datetime):
    if date.month != today.month:
        if date.year == today.year:
            return Filled.YEAR
        else:
            return Filled.NONE
    else:
        if date.year != today.year:
            return Filled.MONTH
        else:
            return Filled.BOTH
    pass


# Enum to keep track of state
class Found(Enum):
    NEITHER = 1
    EXP = 2
    EDU = 3
    BOTH = 4


# Takes in raw resume text
def predict_yoe(resume_text: str) -> dict:
    debug_log = {}
    TODAY = datetime.today()

    # Set indices for common sections of resume
    experience_index = -1
    education_index = -1
    resume_text = clean_text(resume_text)

    # Find present keyword
    try:
        present_indices = [m.start() for m in re.finditer('present', resume_text.lower())]
        today = datetime.today().strftime('%B %d %Y')
        src_str = re.compile('present', re.IGNORECASE)
        resume_text = src_str.sub(today, resume_text)
        debug_log['present'] = True
    except ValueError:
        debug_log['present'] = False
        print('Present not found')

    # Load ner
    doc = nlp(resume_text)

    date_positions = [ent.start_char for ent in doc.ents if ent.type == 'DATE']

    # Locate experience and education blocks
    try:
        experience_indices = [m.start() for m in re.finditer('experience', resume_text.lower())]
        experience_index = weighted_keyword(experience_indices, date_positions)
    except ValueError:
        print("Could not find experience index")

    try:
        education_indices = [m.start() for m in re.finditer('education', resume_text.lower())]
        education_index = weighted_keyword(education_indices, date_positions)
    except ValueError:
        print("Could not find education index")

    debug_log['exp_ind'] = experience_index
    debug_log['edu_ind'] = education_index

    state = None

    # Set state
    if experience_index == education_index == -1:
        state = Found.NEITHER
    elif experience_index != -1 and education_index != -1:
        state = Found.BOTH
    elif experience_index == -1:
        state = Found.EDU
    else:
        state = Found.EXP

    debug_log['state'] = state

    # Isolate chunk of experience text
    if state == Found.BOTH:
        experience_text = resume_text[
                          experience_index:education_index] if experience_index < education_index else resume_text[
                                                                                                       experience_index:]
        education_text = resume_text[
                         education_index:] if experience_index < education_index else resume_text[
                                                                                      education_index:experience_index]

    elif state == Found.EXP:
        experience_text = resume_text[experience_index:]
        education_text = resume_text[:experience_index]

    elif state == Found.EDU:
        if majority_partition(education_index, date_positions) < 0:
            experience_text = resume_text[:education_index]
            education_text = resume_text[education_index:]
        elif majority_partition(education_index, date_positions) > 0:
            experience_text = resume_text[education_index:]
            education_text = resume_text[:education_index]

    else:
        experience_text = resume_text
        education_text = ""

    debug_log['exp_txt'] = experience_text
    debug_log['edu_txt'] = education_text

    # reload model with only education text
    doc = nlp(education_text)

    education_dates = [dateparser.parse(ent.text) for ent in doc.ents if
                       ent.type == 'DATE' and dateparser.parse(ent.text) is not None]
    end_education_date = None if len(education_dates) == 0 else max(education_dates)
    if end_education_date:
        max_bound = (datetime.today() - end_education_date).total_seconds() / SECONDS_IN_YEAR
        if max_bound == 0:
            max_bound = None
        debug_log['max_bound'] = max_bound
    else:
        max_bound = 10000

    debug_log['end_education_date'] = end_education_date

    # Reload model with only experience text
    doc = nlp(experience_text)

    # Recursively reevaluate and parse "None" dates
    # recursive_ent_list = []
    # for ent in doc.ents:
    #     if ent.type == 'DATE' and dateparser.parse(ent.text) is None:
    #         doc2 = nlp(ent.text)
    #         ent_list = [{
    #             'start_char': ent2.start_char + ent.start_char,
    #             'end_char': ent2.end_char + ent.end_char,
    #             'text': ent2.text,
    #             'type': ent2.type
    #         } for ent2 in doc2.ents]
    #         recursive_ent_list += ent_list

    # Index lookup
    date_positions = [ent.start_char for ent in doc.ents if ent.type == 'DATE']
    # date_positions += [ent['start_char'] for ent in recursive_ent_list]
    org_positions = [ent.start_char for ent in doc.ents if ent.type == 'ORG']
    org_start_char_dict = {ent.start_char: ent.text for ent in doc.ents if ent.type == 'ORG'}
    date_start_char_dict = {ent.start_char: ent.text for ent in doc.ents if ent.type == 'DATE'}
    # date_start_char_dict.update({ent['start_char']: ent['text'] for ent in recursive_ent_list})
    date_org_dict = zipped_interval_overlap(date_positions, org_positions)

    res = [{
        'raw_date': date_start_char_dict[start_char],
        'parsed_date': dateparser.parse(date_start_char_dict[start_char]),
        'relevant_orgs': [org_start_char_dict[pos] for pos in org_arr],
        'proximity_dates': [dateparser.parse(date_start_char_dict[pos]) for pos in date_positions if
                            start_char > pos != start_char and
                            start_char - pos <= min_interval(date_positions) and
                            dateparser.parse(date_start_char_dict[pos]) is not None],
        'fill': '',  # Will be defined later
        'confidence': 0
    } for start_char, org_arr in date_org_dict.items()]

    # Calculate Confidence
    for date in res:
        date['confidence'] += len(date['relevant_orgs'])

    res = [date for date in res if date['parsed_date'] is not None]

    # Add in fill status of dates
    for date in res:
        date['fill'] = date_fill(date['parsed_date'], TODAY)

    debug_log['all_extracted_dates'] = res

    # Eliminate dates without proximity dates and NONE
    backup = res.copy()

    # Partition into proximity dates pairs
    res = [[date, date['proximity_dates'][0]] for date in res if
           len(date['proximity_dates']) > 0 and date['parsed_date'] is not None]

    # Correct back-filled values
    for date_pair in res:
        if date_pair[0]['fill'] == Filled.YEAR:
            date_pair[0]['parsed_date'] = date_pair[0]['parsed_date'].replace(date_pair[1].year,
                                                                              date_pair[0]['parsed_date'].month,
                                                                              date_pair[0]['parsed_date'].day)

    res = [[date[0]['parsed_date'], date[1]] for date in res]

    if len(res) > 0:
        res = [[d1, d2] if d1 < d2 else [d2, d1] for [d1, d2] in res]
        res = sorted(res, key=lambda date_pair: date_pair[0])
    else:
        res = backup
        debug_log['backup'] = True
        res = [date['parsed_date'] for date in res if date['confidence'] > 0]
        if len(res) > 0:
            years = (max(res) - (
                min(res) if end_education_date is None else end_education_date)).total_seconds() / SECONDS_IN_YEAR
            return {
                'years': min([years, max_bound]),
                'debug_log': debug_log
            }
        else:
            res = backup
            res = [date['parsed_date'] for date in res]
            years = (max(res) - (
                min(res) if end_education_date is None else end_education_date)).total_seconds() / SECONDS_IN_YEAR
            return {
                'years': min([years, max_bound]),
                'debug_log': debug_log
            }

    debug_log['trimmed_extracted'] = res

    # Date interval reduction
    reduced = []
    recall = 1
    while recall != 0:
        recall -= 1
        for i in range(len(res) - 1):
            if res[i][1] < res[i + 1][0]:
                reduced.append(res[i])
            elif res[i + 1][1] > res[i][1] >= res[i + 1][0]:
                reduced.append([res[i][0], res[i + 1][1]])
                recall = 1
            elif res[i + 1][1] <= res[i][1]:
                reduced.append(res[i])
                i += 1
                recall = 1
        if recall == 0:
            reduced.append(res[-1])
        res = reduced
        reduced = []

    debug_log['reduced_intervals'] = res

    # Sum time deltas and return result in years
    years = sum([(d2 - d1).total_seconds() for [d1, d2] in res]) / SECONDS_IN_YEAR

    return {
        'years': years,
        'debug_log': debug_log
    }
