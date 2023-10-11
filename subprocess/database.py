import json


FILE_PATH = "user_data.json"



def save_users(users):
    with open(FILE_PATH, "w") as fp:
        json.dump(users, fp)

def add_user(email: str, password: str):
    users = get_users()
    users.setdefault(email, password)
    save_users(users)

def authenticate(email: str, password: str) -> bool:
    users = get_users()
    return users.get(email) == password
