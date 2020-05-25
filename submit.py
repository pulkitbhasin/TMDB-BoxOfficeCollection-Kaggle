import os
import re
import json
import requests
import time

SCRIPT_URL = "https://script.google.com/a/berkeley.edu/macros/s/AKfycbx2-mbQMEUgEof_q5yJkB23Ij5Bd_6Hu-tiMeLnZaoN-P8VNXnR/exec"

def setup_kaggle(kaggle_user_data):
    os.system('pip install kaggle')
    os.system('chmod 600 ~/.kaggle/kaggle.json')
    os.system("echo '{0}' > ~/.kaggle/kaggle.json".format(json.dumps(kaggle_user_data)))

def upload_submission(kaggle_user_data, path_to_submission, submission_msg):
    setup_kaggle(kaggle_user_data)
    if submission_msg is None:
        submission_msg = input("Enter a commit message for your submission: ")
    message = os.popen(
            'kaggle competitions submit -c tmdb-box-office-prediction -f {0} -m "{1}"'.format(
                path_to_submission, submission_msg
            )
        ).read()
    if 'success' not in message.lower():
        print('''Unsuccessful submission. Things to check:
        - run "!which kaggle" and make sure you see output
        - run "!cat ~/.kaggle/kaggle.json" and make sure it has your username and key
        - Make sure that you hit the blue button on the competition page and accepted the agreement
        ''')
        return False
    return True

def submit_to_leaderboard(kaggle_user_data, team_name, path_to_submission=None, submission_msg=None, submit_to_kaggle=True):
    setup_kaggle(kaggle_user_data)
    kaggle_submission_success = True
    if submit_to_kaggle:
        print('Submitting to Kaggle...')
        kaggle_submission_success = upload_submission(kaggle_user_data, path_to_submission, submission_msg)
    if kaggle_submission_success:
        print('Reading Kaggle submissions...')
        tries = 0
        while tries < 5:
            submissions = os.popen(
                'kaggle competitions submissions -c tmdb-box-office-prediction'
            ).read()
            if submissions:
                lines = submissions.strip().split('\n')
                scores_line = None
                for i, line in enumerate(lines):
                    if line.startswith('---'):
                        scores_line = lines[i + 1]
                        break
                if scores_line is None:
                    break
                score_str = re.split('\s+', scores_line.strip())[-2]
                if score_str is not None and score_str[0].isdigit():
                    score = float(score_str)
                    return push_score_to_leaderboard(team_name, score)
            time.sleep(1)
            tries += 1
        print('''No scores found for recent submission, which either means that Kaggle is processing your submission, you tried too many times to submit, or something else went wrong. Wait an hour (you can check the status of your submission by running view_submissions(KAGGLE_USER_DATA)), and run the submit function with the flag submit_to_kaggle=False, and if that doesn't work, contact us.''')
    return False

def view_submissions(kaggle_user_data):
    setup_kaggle(kaggle_user_data)
    submissions = os.popen(
        'kaggle competitions submissions -c tmdb-box-office-prediction'
    ).read()
    return submissions

def push_score_to_leaderboard(team_name, score):
    print('Pushing score {0} for team {1}'.format(score, team_name))
    r = requests.post(
        SCRIPT_URL,
        data={
            'team_name': team_name,
            'score': score
        }
    )
    if not r.ok:
        print('Failure. Are you connected to the internet?')
        return False
    print('Success!')
    return True