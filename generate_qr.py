import pyotp, qrcode, sqlite3

# Step 1: Generate a secret for the user
otp_secret = pyotp.random_base32()

# Step 2: Save it in your SQLite users table
conn = sqlite3.connect("database/financial_aid.db")
cursor = conn.cursor()
cursor.execute("UPDATE users SET otp_secret=? WHERE email=?", (otp_secret, "user@example.com"))
conn.commit()
conn.close()

# Step 3: Generate QR code
totp = pyotp.TOTP(otp_secret)
qr_uri = totp.provisioning_uri(name="user@example.com", issuer_name="FinancialAidSystem")
img = qrcode.make(qr_uri)
img.save("otp_qr.png")

print("QR code saved as otp_qr.png — scan it in Google Authenticator!")
