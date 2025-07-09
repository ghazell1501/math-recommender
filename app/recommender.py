import pandas as pd
from recommendation_list import RECOMMENDATION_LIST

def load_data(student_json):
    '''
    Takes student's scores in JSON format as input, and returns the following:
        1. seen (DataFrame) - questions the student has "studied in depth" or "studied but forgotten"
        2. unseen (DataFrame) - questions the student has "never seen before"
        3. total (DataFrame) - all questions with their metadata
        4. number_of_questions (int) - total number of questions
        5. score (int) - number of correctly answered questions
    '''

    # Convert JSON to DataFrame
    all_questions = next(iter(student_json.values()))
    all_rows = []
    for question_id, info in all_questions.items():
        row = {
            "question_id": question_id,
            "Score": info["Score"],
            "Concept": info["Concept"],
            "Topic": info["Topic"],
            "Familiarity": info["Familiarity"]
        }
        all_rows.append(row)

    # Create separate DataFrames for seen and unseen questions 
    total = pd.DataFrame(all_rows)
    number_of_questions = total.shape[0]
    score = total["Score"].sum()

    seen = total[total["Familiarity"].isin(["Studied in depth", "Studied but forgotten"])]
    unseen = total[total["Familiarity"] == "Never Studied"]

    return seen, unseen, total, number_of_questions, score

def get_recommendations(seen, unseen, total, number_of_questions):
    '''
    Returns personalized recommendations for seen and unseen topics:
        1. recommendations_seen (DataFrame) - for concepts previously encountered but misunderstood
        2. recommendations_unseen (DataFrame) - for concepts never studied
    '''

    # Load list of resources for recommendation 
    recommendations_list = RECOMMENDATION_LIST

    ''' Seen topics'''
    # Calculate number of mistakes and total questions per topic 
    seen_ordered = seen.groupby('Topic').agg(
        mistakes=('Score', lambda x: (x == 0).sum()),
        total_questions=('Score', 'count')
    )
    
    # Calculate priority score for each topic based on fraction of mistakes and fraction of questions, and order by highest priority first 
    seen_ordered['fraction_mistakes'] = seen_ordered['mistakes'] / seen_ordered['total_questions']
    seen_ordered['fraction_questions'] = seen_ordered['total_questions'] / number_of_questions

    seen_ordered['priority'] = (
        0.5 * seen_ordered['fraction_mistakes'] +
        0.5 * seen_ordered['fraction_questions']
    )
    seen_ordered['priority'] = seen_ordered['priority'] / seen_ordered['priority'].sum()
    seen_ordered = seen_ordered.sort_values(by='priority', ascending=False).reset_index()

    # Find all the concepts where mistakes were made, per topic, in the order of priority calculated above
    recommendations_seen = []
    for topic in seen_ordered['Topic']:
        wrong_concepts = total[(total['Topic'] == topic) & (total['Score'] == 0)][['Concept']]
        wrong_concepts['Concept'] = wrong_concepts['Concept'].str.strip().str.lower()
        recommendations_list['Concept'] = recommendations_list['Concept'].str.strip().str.lower()

        # Find resources for wrong concepts from the recommendation list, and order them by order in which concepts should be studied 
        ordered = pd.merge(wrong_concepts, recommendations_list, on='Concept', how='left') 
        ordered = ordered.sort_values(by='Order')
        for _, row in ordered.iterrows():
            recommendations_seen.append({
                'Concept': row['Concept'],
                'Link': row['Link'],
                'Topic':row['Topic']
            })
    
    # Remove duplicate concepts, test can have multiple questions from the same concept, but we want to recommend each concept only once 
    unique_seen = {}
    for item in recommendations_seen:
        unique_seen[item['Concept']] = item 
    recommendations_seen = pd.DataFrame(unique_seen.values())

    ''' Unseen topics'''
    # Prioritize unseen topics by number of questions per topic, and order by highest number of questions first 
    unseen_ordered = unseen.groupby('Topic').agg(
        total_questions=('Score', 'count')
    ).sort_values(by='total_questions', ascending=False).reset_index()

    # Find all the concepts where mistakes were made, per topic, in the order of priority calculated above
    recommendations_unseen = []
    for topic in unseen_ordered['Topic']:
        wrong_concepts = total[(total['Topic'] == topic) & (total['Score'] == 0)][['Concept']]

        wrong_concepts['Concept'] = wrong_concepts['Concept'].str.strip().str.lower()
        recommendations_list['Concept'] = recommendations_list['Concept'].str.strip().str.lower()

        # Find resources for wrong concepts from the recommendation list, and order them by order in which concepts should be studied
        ordered = pd.merge(wrong_concepts, recommendations_list, on='Concept', how='left')
        ordered = ordered.sort_values(by='Order')
        for _, row in ordered.iterrows():
            recommendations_unseen.append({
                'Concept': row['Concept'],
                'Link': row['Link'], 
                'Topic': row['Topic']
            })

    # Remove duplicate concepts, test can have multiple questions from the same concept, but we want to recommend each concept only once 
    unique_unseen = {}
    for item in recommendations_unseen:
        unique_unseen[item['Concept']] = item
    recommendations_unseen = pd.DataFrame(unique_unseen.values())

    return recommendations_seen, recommendations_unseen