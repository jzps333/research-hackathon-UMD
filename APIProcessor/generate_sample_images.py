"""
Helper script to generate local sample images for multimodal testing.
Creates realistic images in the 'images/' folder representing each test row.
"""

import os
from PIL import Image, ImageDraw, ImageFont


def create_sample_images():
    os.makedirs("images", exist_ok=True)

    # 1. USPS Smishing SMS Screenshot
    img1 = Image.new("RGB", (600, 300), color=(245, 245, 247))
    d1 = ImageDraw.Draw(img1)
    # Header bar
    d1.rectangle([(0, 0), (600, 50)], fill=(230, 230, 235))
    d1.text((20, 15), "Messages: +1 (833) 492-0199 (Unknown Sender)", fill=(50, 50, 50))
    # SMS Bubble
    d1.rounded_rectangle([(30, 80), (570, 230)], radius=15, fill=(229, 229, 234))
    sms_text = (
        "[USPS Alert]: Package #US9514901192 is on hold\n"
        "due to missing street address.\n\n"
        "Please confirm your delivery address at:\n"
        "http://usps-redelivery-portal.top/track\n"
        "Action required within 12 hours or item returns."
    )
    d1.text((50, 95), sms_text, fill=(0, 0, 0), spacing=6)
    img1.save("images/usps_smishing.png")

    # 2. Recipe Card
    img2 = Image.new("RGB", (600, 350), color=(253, 246, 227))
    d2 = ImageDraw.Draw(img2)
    d2.rectangle([(10, 10), (590, 340)], outline=(180, 160, 120), width=2)
    d2.text((40, 30), "RECIPE: Rustic Potato Leek Soup", fill=(80, 50, 20))
    recipe_text = (
        "- 4 large leeks, washed & sliced\n"
        "- 3 russet potatoes, peeled & cubed\n"
        "- 4 cups vegetable broth\n"
        "- 1 cup heavy cream\n"
        "- 2 tbsp butter, salt & fresh pepper\n\n"
        "Instructions: Simmer leeks in butter for 10 min.\n"
        "Add potatoes and broth. Simmer for 25 min until tender."
    )
    d2.text((40, 70), recipe_text, fill=(70, 60, 50), spacing=8)
    img2.save("images/recipe_card.png")

    # 3. Telegram Job Offer Chat Screenshot
    img3 = Image.new("RGB", (600, 340), color=(240, 242, 245))
    d3 = ImageDraw.Draw(img3)
    d3.rectangle([(0, 0), (600, 45)], fill=(84, 130, 237))
    d3.text((20, 12), "Telegram - Global Logistics HR Director", fill=(255, 255, 255))
    d3.rounded_rectangle([(30, 70), (570, 280)], radius=12, fill=(255, 255, 255), outline=(210, 215, 225))
    job_text = (
        "HR Coordinator (10:14 AM):\n"
        "Congratulations! You are selected for the Remote Data Entry Specialist.\n"
        "Pay rate: $55.00/hour (Weekly Direct Deposit).\n\n"
        "To set up your home office workstation:\n"
        "1. We will issue you a certified company check for $3,200.\n"
        "2. Deposit check at your bank ATM.\n"
        "3. Once deposited, send $2,400 via Zelle to our vendor today for equipment."
    )
    d3.text((45, 85), job_text, fill=(30, 30, 30), spacing=6)
    img3.save("images/telegram_job.png")

    # 4. Washing Machine Rear Panel Diagram
    img4 = Image.new("RGB", (600, 320), color=(248, 249, 250))
    d4 = ImageDraw.Draw(img4)
    d4.rectangle([(15, 15), (585, 305)], outline=(100, 110, 120), width=2)
    d4.text((30, 25), "TECHNICAL DIAGRAM - REAR HOUSING & INLET VALVES", fill=(30, 40, 50))
    d4.rectangle([(50, 70), (220, 250)], fill=(220, 225, 230), outline=(150, 150, 150))
    d4.text((70, 110), "Cold Water Inlet\n(3/4\" Hose Thread)", fill=(20, 40, 160))
    d4.text((70, 180), "Hot Water Inlet\n(3/4\" Hose Thread)", fill=(180, 40, 20))
    d4.rectangle([(280, 70), (550, 250)], fill=(235, 240, 245), outline=(150, 150, 150))
    d4.text((300, 90), "Maintenance & Specs:\n- Internal mesh filter screen inside cold valve\n- Replacement hose: Standard 3/4\" FHT x 3/4\" FHT\n- Max operating pressure: 120 PSI\n- Clean sediment screen every 6 months", fill=(40, 50, 60), spacing=6)
    img4.save("images/appliance_valve.png")

    # 5. Fake Invoice / Renewal Notice
    img5 = Image.new("RGB", (600, 330), color=(255, 255, 255))
    d5 = ImageDraw.Draw(img5)
    d5.rectangle([(0, 0), (600, 50)], fill=(34, 45, 65))
    d5.text((20, 15), "INVOICE & AUTO-RENEWAL RECEIPT", fill=(255, 255, 255))
    d5.text((30, 70), "Billed To: Corporate Client Services", fill=(80, 80, 80))
    d5.text((30, 95), "Service: Geek Cloud Protection 3-Year Extended License", fill=(0, 0, 0))
    d5.text((30, 125), "Amount Debited: $799.00 USD (Auto-Debit from primary card)", fill=(180, 0, 0))
    d5.rounded_rectangle([(30, 170), (570, 270)], radius=8, fill=(254, 242, 242), outline=(239, 68, 68))
    notice_text = (
        "IMPORTANT NOTICE:\n"
        "If you did not authorize this charge or wish to cancel subscription,\n"
        "do NOT email. Call our Billing Support line immediately:\n"
        "TOLL FREE: 1-888-555-0194 (24/7 Dispute Team)"
    )
    d5.text((45, 185), notice_text, fill=(153, 27, 27), spacing=6)
    img5.save("images/fake_invoice.png")

    print("[SUCCESS] Created 5 sample images in 'images/' directory.")


if __name__ == "__main__":
    create_sample_images()
