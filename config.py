# config.py
USER_CREDENTIALS = {
    "qwerty": "qwerty",
    "qwerty": "Mustu@123"
    
}

# Function to hash a password
def hash_password(password):
    import hashlib
    return hashlib.sha256(password.encode()).hexdigest()
