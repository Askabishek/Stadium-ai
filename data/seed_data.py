"""
Synthetic data generator for StadiumAI — FIFA World Cup 2026 venues.
Generates realistic stadium data, events, crowd info, and facility details.
"""

import sqlite3
import random
import os
from datetime import datetime, timedelta

DB_PATH = os.path.join(os.path.dirname(__file__), "stadium_db.sqlite")

# FIFA World Cup 2026 Venues (USA, Mexico, Canada)
STADIUMS = [
    {
        "name": "MetLife Stadium",
        "city": "East Rutherford, NJ",
        "country": "USA",
        "capacity": 82500,
        "lat": 40.8128,
        "lon": -74.0742
    },
    {
        "name": "AT&T Stadium",
        "city": "Arlington, TX",
        "country": "USA",
        "capacity": 80000,
        "lat": 32.7473,
        "lon": -97.0945
    },
    {
        "name": "SoFi Stadium",
        "city": "Inglewood, CA",
        "country": "USA",
        "capacity": 70240,
        "lat": 33.9535,
        "lon": -118.3392
    },
    {
        "name": "Estadio Azteca",
        "city": "Mexico City",
        "country": "Mexico",
        "capacity": 87523,
        "lat": 19.3029,
        "lon": -99.1505
    },
    {
        "name": "Hard Rock Stadium",
        "city": "Miami, FL",
        "country": "USA",
        "capacity": 64767,
        "lat": 25.9580,
        "lon": -80.2389
    },
    {
        "name": "Lumen Field",
        "city": "Seattle, WA",
        "country": "USA",
        "capacity": 68740,
        "lat": 47.5952,
        "lon": -122.3316
    },
    {
        "name": "BMO Field",
        "city": "Toronto",
        "country": "Canada",
        "capacity": 45736,
        "lat": 43.6332,
        "lon": -79.4186
    },
    {
        "name": "NRG Stadium",
        "city": "Houston, TX",
        "country": "USA",
        "capacity": 72220,
        "lat": 29.6847,
        "lon": -95.4107
    },
]

FACILITIES = [
    {"type": "Restroom", "icon": "🚻"},
    {"type": "Food Court", "icon": "🍔"},
    {"type": "First Aid", "icon": "🏥"},
    {"type": "Information Desk", "icon": "ℹ️"},
    {"type": "Merchandise Store", "icon": "🛍️"},
    {"type": "ATM", "icon": "🏧"},
    {"type": "Prayer Room", "icon": "🕌"},
    {"type": "Baby Care Room", "icon": "👶"},
    {"type": "Wheelchair Access Point", "icon": "♿"},
    {"type": "Water Station", "icon": "💧"},
    {"type": "Charging Station", "icon": "🔋"},
    {"type": "Lost & Found", "icon": "📦"},
    {"type": "Exit Gate", "icon": "🚪"},
    {"type": "Parking Shuttle", "icon": "🚌"},
    {"type": "VIP Lounge", "icon": "⭐"},
]

ZONES = ["North Stand", "South Stand", "East Wing", "West Wing", 
         "VIP Section", "Family Zone", "General Area", "Concourse Level 1",
         "Concourse Level 2", "Gate A", "Gate B", "Gate C", "Gate D"]

LANGUAGES = ["English", "Spanish", "French", "Arabic", "Hindi", 
             "Portuguese", "German", "Japanese", "Korean", "Mandarin"]

CROWD_DENSITY_LEVELS = ["Low", "Moderate", "High", "Very High", "Critical"]

ANNOUNCEMENT_TYPES = ["Safety", "Match Update", "Navigation", "Weather", 
                      "Schedule Change", "Emergency", "General"]

SAMPLE_ANNOUNCEMENTS = [
    "Match kickoff delayed by 15 minutes due to weather conditions.",
    "Gate C is experiencing high crowd density. Please use Gate D for faster entry.",
    "Lost child reported near Section 204. Please contact nearest security.",
    "Water stations available at all concourse levels. Stay hydrated!",
    "Shuttle buses to parking lot B depart every 10 minutes from Gate A.",
    "Half-time entertainment begins in 5 minutes at the North Stand.",
    "Emergency exits are located at all four corners of the stadium.",
    "Free WiFi available: Connect to FIFA_WC2026_Guest network.",
    "Wheelchair assistance available at Information Desks on all levels.",
    "Fireworks display after the match. Please remain seated until cleared.",
]

TRANSPORT_OPTIONS = [
    {"mode": "Metro/Subway", "details": "Line 4 to Stadium Station, 5 min walk"},
    {"mode": "Shuttle Bus", "details": "Free shuttle from City Center every 15 min"},
    {"mode": "Ride Share", "details": "Designated pickup/dropoff at Lot E"},
    {"mode": "Walking", "details": "Pedestrian route from downtown, 20 min"},
    {"mode": "Bicycle", "details": "Bike parking available at Gate B"},
    {"mode": "Taxi", "details": "Taxi rank at South entrance"},
    {"mode": "Private Car", "details": "Parking lots A-D, pre-booking required"},
]


def create_database():
    """Create the SQLite database and all tables."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS stadiums (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            city TEXT NOT NULL,
            country TEXT NOT NULL,
            capacity INTEGER NOT NULL,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS facilities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stadium_id INTEGER NOT NULL,
            facility_type TEXT NOT NULL,
            icon TEXT,
            zone TEXT NOT NULL,
            floor_level INTEGER DEFAULT 1,
            is_accessible INTEGER DEFAULT 1,
            latitude REAL,
            longitude REAL,
            description TEXT,
            FOREIGN KEY (stadium_id) REFERENCES stadiums(id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS crowd_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stadium_id INTEGER NOT NULL,
            zone TEXT NOT NULL,
            density_level TEXT NOT NULL,
            estimated_count INTEGER,
            max_capacity INTEGER,
            recorded_at TIMESTAMP NOT NULL,
            FOREIGN KEY (stadium_id) REFERENCES stadiums(id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS announcements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stadium_id INTEGER NOT NULL,
            announcement_type TEXT NOT NULL,
            message TEXT NOT NULL,
            language TEXT DEFAULT 'English',
            priority INTEGER DEFAULT 1,
            created_at TIMESTAMP NOT NULL,
            is_active INTEGER DEFAULT 1,
            FOREIGN KEY (stadium_id) REFERENCES stadiums(id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS matches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stadium_id INTEGER NOT NULL,
            team_a TEXT NOT NULL,
            team_b TEXT NOT NULL,
            match_date TIMESTAMP NOT NULL,
            match_type TEXT NOT NULL,
            status TEXT DEFAULT 'Scheduled',
            FOREIGN KEY (stadium_id) REFERENCES stadiums(id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS transport (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stadium_id INTEGER NOT NULL,
            mode TEXT NOT NULL,
            details TEXT NOT NULL,
            is_accessible INTEGER DEFAULT 1,
            estimated_time_min INTEGER,
            cost_estimate TEXT,
            FOREIGN KEY (stadium_id) REFERENCES stadiums(id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stadium_id INTEGER NOT NULL,
            category TEXT NOT NULL,
            message TEXT NOT NULL,
            sentiment TEXT,
            language TEXT DEFAULT 'English',
            created_at TIMESTAMP NOT NULL,
            FOREIGN KEY (stadium_id) REFERENCES stadiums(id)
        )
    """)

    conn.commit()
    conn.close()


def seed_stadiums():
    """Seed stadium data."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    for s in STADIUMS:
        cursor.execute("""
            INSERT INTO stadiums (name, city, country, capacity, latitude, longitude)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (s["name"], s["city"], s["country"], s["capacity"], s["lat"], s["lon"]))
    conn.commit()
    conn.close()


def seed_facilities():
    """Seed facility data for each stadium."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    for stadium_id in range(1, len(STADIUMS) + 1):
        stadium = STADIUMS[stadium_id - 1]
        for facility in FACILITIES:
            num_instances = random.randint(2, 6)
            for _ in range(num_instances):
                zone = random.choice(ZONES)
                floor_level = random.randint(1, 3)
                is_accessible = random.choices([1, 0], weights=[80, 20])[0]
                lat = stadium["lat"] + random.uniform(-0.002, 0.002)
                lon = stadium["lon"] + random.uniform(-0.002, 0.002)
                desc = f"{facility['type']} located in {zone}, Level {floor_level}"

                cursor.execute("""
                    INSERT INTO facilities 
                    (stadium_id, facility_type, icon, zone, floor_level, 
                     is_accessible, latitude, longitude, description)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (stadium_id, facility["type"], facility["icon"],
                      zone, floor_level, is_accessible, lat, lon, desc))

    conn.commit()
    conn.close()


def seed_crowd_data():
    """Seed crowd density data."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    now = datetime.now()

    for stadium_id in range(1, len(STADIUMS) + 1):
        for zone in ZONES:
            for hours_ago in range(0, 48, 2):
                density = random.choice(CROWD_DENSITY_LEVELS)
                max_cap = random.randint(3000, 15000)
                density_pcts = {"Low": 0.2, "Moderate": 0.4, "High": 0.6,
                                "Very High": 0.8, "Critical": 0.95}
                estimated = int(max_cap * density_pcts.get(density, 0.5))
                recorded_at = now - timedelta(hours=hours_ago)

                cursor.execute("""
                    INSERT INTO crowd_data 
                    (stadium_id, zone, density_level, estimated_count, 
                     max_capacity, recorded_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (stadium_id, zone, density, estimated, max_cap,
                      recorded_at.strftime("%Y-%m-%d %H:%M:%S")))

    conn.commit()
    conn.close()


def seed_announcements():
    """Seed announcement data."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    now = datetime.now()

    for stadium_id in range(1, len(STADIUMS) + 1):
        for i in range(15):
            ann_type = random.choice(ANNOUNCEMENT_TYPES)
            message = random.choice(SAMPLE_ANNOUNCEMENTS)
            language = random.choice(LANGUAGES[:5])
            priority = random.randint(1, 5)
            hours_ago = random.randint(0, 72)
            created_at = now - timedelta(hours=hours_ago)

            cursor.execute("""
                INSERT INTO announcements 
                (stadium_id, announcement_type, message, language, 
                 priority, created_at, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (stadium_id, ann_type, message, language, priority,
                  created_at.strftime("%Y-%m-%d %H:%M:%S"),
                  1 if hours_ago < 24 else 0))

    conn.commit()
    conn.close()


def seed_matches():
    """Seed match schedule data."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    teams = ["Brazil", "Argentina", "Germany", "France", "Spain", "England",
             "Portugal", "Netherlands", "Italy", "USA", "Mexico", "Canada",
             "Japan", "South Korea", "Australia", "Morocco", "Senegal", "Nigeria"]
    match_types = ["Group Stage", "Round of 16", "Quarter Final", 
                   "Semi Final", "Final"]

    base_date = datetime(2026, 6, 11)

    for stadium_id in range(1, len(STADIUMS) + 1):
        for i in range(5):
            team_a, team_b = random.sample(teams, 2)
            match_date = base_date + timedelta(days=random.randint(0, 45))
            match_type = random.choice(match_types)
            status = random.choice(["Scheduled", "Live", "Completed"])

            cursor.execute("""
                INSERT INTO matches 
                (stadium_id, team_a, team_b, match_date, match_type, status)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (stadium_id, team_a, team_b,
                  match_date.strftime("%Y-%m-%d %H:%M:%S"), match_type, status))

    conn.commit()
    conn.close()


def seed_transport():
    """Seed transport data for each stadium."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    for stadium_id in range(1, len(STADIUMS) + 1):
        for transport in TRANSPORT_OPTIONS:
            is_accessible = random.choices([1, 0], weights=[70, 30])[0]
            est_time = random.randint(5, 45)
            costs = ["Free", "$5", "$10", "$15", "$20", "$25-40"]
            cost = random.choice(costs)

            cursor.execute("""
                INSERT INTO transport 
                (stadium_id, mode, details, is_accessible, 
                 estimated_time_min, cost_estimate)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (stadium_id, transport["mode"], transport["details"],
                  is_accessible, est_time, cost))

    conn.commit()
    conn.close()


def seed_feedback():
    """Seed sample feedback data."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    now = datetime.now()

    categories = ["Navigation", "Food", "Cleanliness", "Staff", 
                  "Accessibility", "Safety", "Transport", "General"]
    sentiments = ["Positive", "Neutral", "Negative"]
    
    feedback_samples = [
        "Great experience finding my seat with the AI assistant!",
        "Food court was too crowded, need better queue management.",
        "Wheelchair ramps were easy to find. Very accessible!",
        "Staff was very helpful when I got lost.",
        "The shuttle service was confusing. Needs better signage.",
        "Loved the multilingual announcements!",
        "Restrooms were clean and well-maintained.",
        "Parking was a nightmare. Took 45 minutes to find a spot.",
        "The AI navigation helped me find the nearest first aid quickly.",
        "Would be great to have real-time crowd updates on the app.",
    ]

    for stadium_id in range(1, len(STADIUMS) + 1):
        for i in range(20):
            category = random.choice(categories)
            message = random.choice(feedback_samples)
            sentiment = random.choices(sentiments, weights=[50, 30, 20])[0]
            language = random.choice(LANGUAGES[:5])
            hours_ago = random.randint(0, 168)
            created_at = now - timedelta(hours=hours_ago)

            cursor.execute("""
                INSERT INTO feedback 
                (stadium_id, category, message, sentiment, language, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (stadium_id, category, message, sentiment, language,
                  created_at.strftime("%Y-%m-%d %H:%M:%S")))

    conn.commit()
    conn.close()


def initialize_database():
    """Initialize database with seed data if empty."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM stadiums")
    count = cursor.fetchone()[0]
    conn.close()

    if count == 0:
        print("🏟️ Seeding StadiumAI database...")
        seed_stadiums()
        seed_facilities()
        seed_crowd_data()
        seed_announcements()
        seed_matches()
        seed_transport()
        seed_feedback()
        print("✅ Database seeded successfully!")
    else:
        print(f"📊 Database already has {count} stadiums loaded.")


if __name__ == "__main__":
    create_database()
    initialize_database()
