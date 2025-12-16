import json
import random
import argparse
from datetime import datetime, timedelta

# Expanded knowledge base with over 200 facts
FACTS = {
    "Afghanistan": "Kabul", "Albania": "Tirana (Tirane)", "Algeria": "Algiers",
    "Andorra": "Andorra la Vella", "Angola": "Luanda", "Antigua and Barbuda": "Saint John's",
    "Argentina": "Buenos Aires", "Armenia": "Yerevan", "Australia": "Canberra",
    "Austria": "Vienna", "Azerbaijan": "Baku", "Bahamas": "Nassau",
    "Bahrain": "Manama", "Bangladesh": "Dhaka", "Barbados": "Bridgetown",
    "Belarus": "Minsk", "Belgium": "Brussels", "Belize": "Belmopan",
    "Benin": "Porto Novo", "Bhutan": "Thimphu", "Bolivia": "Sucre",
    "Bosnia and Herzegovina": "Sarajevo", "Botswana": "Gaborone", "Brazil": "Brasilia",
    "Brunei": "Bandar Seri Begawan", "Bulgaria": "Sofia", "Burkina Faso": "Ouagadougou",
    "Burundi": "Gitega", "Cambodia": "Phnom Penh", "Cameroon": "Yaounde",
    "Canada": "Ottawa", "Cape Verde": "Praia", "Central African Republic": "Bangui",
    "Chad": "N'Djamena", "Chile": "Santiago", "China": "Beijing",
    "Colombia": "Bogota", "Comoros": "Moroni", "Congo, Democratic Republic of the": "Kinshasa",
    "Congo, Republic of the": "Brazzaville", "Costa Rica": "San Jose",
    "Côte d'Ivoire (Ivory Coast)": "Yamoussoukro", "Croatia": "Zagreb", "Cuba": "Havana",
    "Cyprus": "Nicosia", "Czech Republic (Czechia)": "Prague", "Denmark": "Copenhagen",
    "Djibouti": "Djibouti", "Dominica": "Roseau", "Dominican Republic": "Santo Domingo",
    "East Timor": "Dili", "Ecuador": "Quito", "Egypt": "Cairo",
    "El Salvador": "San Salvador", "England": "London", "Equatorial Guinea": "Malabo",
    "Eritrea": "Asmara", "Estonia": "Tallinn", "Eswatini (Swaziland)": "Mbabane",
    "Ethiopia": "Addis Ababa", "Federated States of Micronesia": "Palikir", "Fiji": "Suva",
    "Finland": "Helsinki", "France": "Paris", "Gabon": "Libreville",
    "Gambia": "Banjul", "Georgia": "Tbilisi", "Germany": "Berlin",
    "Ghana": "Accra", "Greece": "Athens", "Grenada": "Saint George's",
    "Guatemala": "Guatemala City", "Guinea": "Conakry", "Guinea-Bissau": "Bissau",
    "Guyana": "Georgetown", "Haiti": "Port au Prince", "Honduras": "Tegucigalpa",
    "Hungary": "Budapest", "Iceland": "Reykjavik", "India": "New Delhi",
    "Indonesia": "Jakarta", "Iran": "Tehran", "Iraq": "Baghdad",
    "Ireland": "Dublin", "Israel": "Jerusalem", "Italy": "Rome",
    "Jamaica": "Kingston", "Japan": "Tokyo", "Jordan": "Amman",
    "Kazakhstan": "Astana", "Kenya": "Nairobi", "Kiribati": "Tarawa Atoll",
    "Kosovo": "Pristina", "Kuwait": "Kuwait City", "Kyrgyzstan": "Bishkek",
    "Laos": "Vientiane", "Latvia": "Riga", "Lebanon": "Beirut",
    "Lesotho": "Maseru", "Liberia": "Monrovia", "Libya": "Tripoli",
    "Liechtenstein": "Vaduz", "Lithuania": "Vilnius", "Luxembourg": "Luxembourg",
    "Madagascar": "Antananarivo", "Malawi": "Lilongwe", "Malaysia": "Kuala Lumpur",
    "Maldives": "Male", "Mali": "Bamako", "Malta": "Valletta",
    "Marshall Islands": "Majuro", "Mauritania": "Nouakchott", "Mauritius": "Port Louis",
    "Mexico": "Mexico City", "Moldova": "Chisinau", "Monaco": "Monaco",
    "Mongolia": "Ulaanbaatar", "Montenegro": "Podgorica", "Morocco": "Rabat",
    "Mozambique": "Maputo", "Myanmar (Burma)": "Nay Pyi Taw", "Namibia": "Windhoek",
    "Nepal": "Kathmandu", "Netherlands": "Amsterdam", "New Zealand": "Wellington",
    "Nicaragua": "Managua", "Niger": "Niamey", "Nigeria": "Abuja",
    "North Korea": "Pyongyang", "North Macedonia (Macedonia)": "Skopje", "Northern Ireland": "Belfast",
    "Norway": "Oslo", "Oman": "Muscat", "Pakistan": "Islamabad",
    "Palau": "Ngerulmud", "Palestine": "Jerusalem", "Panama": "Panama City",
    "Papua New Guinea": "Port Moresby", "Paraguay": "Asuncion", "Peru": "Lima",
    "Philippines": "Manila", "Poland": "Warsaw", "Portugal": "Lisbon",
    "Qatar": "Doha", "Romania": "Bucharest", "Russia": "Moscow",
    "Rwanda": "Kigali", "Saint Kitts and Nevis": "Basseterre", "Saint Lucia": "Castries",
    "Saint Vincent and the Grenadines": "Kingstown", "Samoa": "Apia", "San Marino": "San Marino",
    "Sao Tome and Principe": "Sao Tome", "Saudi Arabia": "Riyadh", "Scotland": "Edinburgh",
    "Senegal": "Dakar", "Serbia": "Belgrade", "Seychelles": "Victoria",
    "Sierra Leone": "Freetown", "Singapore": "Singapore", "Slovakia": "Bratislava",
    "Slovenia": "Ljubljana", "Solomon Islands": "Honiara", "Somalia": "Mogadishu",
    "South Africa": "Pretoria", "South Korea": "Seoul", "South Sudan": "Juba",
    "Spain": "Madrid", "Sri Lanka": "Sri Jayawardenapura Kotte", "Sudan": "Khartoum",
    "Suriname": "Paramaribo", "Sweden": "Stockholm", "Switzerland": "Bern",
    "Syria": "Damascus", "Taiwan": "Taipei", "Tajikistan": "Dushanbe",
    "Tanzania": "Dodoma", "Thailand": "Bangkok", "Togo": "Lome",
    "Tonga": "Nuku'alofa", "Trinidad and Tobago": "Port of Spain", "Tunisia": "Tunis",
    "Türkiye (Turkey)": "Ankara", "Turkmenistan": "Ashgabat", "Tuvalu": "Funafuti",
    "Uganda": "Kampala", "Ukraine": "Kyiv or Kiev", "United Arab Emirates": "Abu Dhabi",
    "United Kingdom": "London", "United States": "Washington D.C.", "Uruguay": "Montevideo",
    "Uzbekistan": "Tashkent", "Vanuatu": "Port Vila", "Vatican City": "Vatican City",
    "Venezuela": "Caracas", "Vietnam": "Hanoi", "Wales": "Cardiff",
    "Yemen": "Sana'a", "Zambia": "Lusaka", "Zimbabwe": "Harare"
}

def generate_stream(num_total_events: int, num_facts: int, es_capacity: int, inject_contradiction: bool = False):
    """
    Generates a long stream of events with facts injected throughout.
    Queries appear after a significant delay to test long-term retention.
    """
    stream = []
    start_time = datetime(2023, 1, 1)

    # 1. Select facts and schedule their injection times
    fact_items = random.sample(list(FACTS.items()), min(num_facts, len(FACTS)))
    
    # Schedule fact injection points randomly across the first half of the stream
    injection_points = sorted(random.sample(range(num_total_events // 2), len(fact_items)))
    
    fact_schedule = {point: fact for point, fact in zip(injection_points, fact_items)}

    # 2. Handle contradiction injection
    contradiction_schedule = {}
    if inject_contradiction and fact_items:
        # Pick a fact to contradict
        contradicted_fact_country, original_capital = fact_items[0]
        
        # Create a believable but incorrect capital
        incorrect_capital = "Berlin" if original_capital != "Berlin" else "Rome" # Just an example
        
        # Schedule the incorrect fact early and the correct one later
        incorrect_injection_point = injection_points[0] // 2
        correct_injection_point = (num_total_events // 2) + (num_total_events // 4) # 75% of the way through
        
        # Update schedules
        fact_schedule[correct_injection_point] = (contradicted_fact_country, original_capital)
        contradiction_schedule[incorrect_injection_point] = (contradicted_fact_country, incorrect_capital)
        
        # Remove the original early injection of the correct fact
        if injection_points[0] in fact_schedule:
            del fact_schedule[injection_points[0]]

    # 3. Generate the event stream
    distractors = [
        "The weather is nice today.", "I'm planning a trip soon.", "Remember to buy groceries.",
        "The stock market is volatile.", "Let's discuss the project timeline.", "My favorite movie is a classic.",
        "I learned a new recipe yesterday.", "The dog needs a walk.", "Planning a weekend hike.",
        "The library books are due soon."
    ]
    
    time_step = timedelta(minutes=5)
    current_time = start_time
    
    for i in range(num_total_events):
        event = {"t": current_time.isoformat()}
        
        if i in contradiction_schedule:
            country, capital = contradiction_schedule[i]
            event["text"] = f"{capital} is the capital of {country}." # Incorrect fact
        elif i in fact_schedule:
            country, capital = fact_schedule[i]
            event["text"] = f"{capital} is the capital of {country}." # Correct fact
        else:
            event["text"] = random.choice(distractors) # Distractor
        
        stream.append(event)
        current_time += time_step

    # 4. Add queries for all facts at the end of the stream
    # This ensures queries happen *after* the ES has likely overflown
    for country, capital in fact_items:
        query_event = {
            "t": (current_time + time_step).isoformat(),
            "query": f"What is the capital of {country}?",
            "answer": capital
        }
        stream.append(query_event)
        current_time += time_step
        
    return stream


def main():
    parser = argparse.ArgumentParser(description="Generate a synthetic episodic QA dataset for memory pressure testing.")
    parser.add_argument("--output_file", type=str, default="data/episodic_qa_pressure.jsonl",
                        help="Path to the output JSONL file.")
    parser.add_argument("--num_events", type=int, default=10000,
                        help="Total number of events (facts + distractors).")
    parser.add_argument("--num_facts", type=int, default=500,
                        help="Number of unique facts to inject.")
    parser.add_argument("--es_capacity", type=int, default=128,
                        help="Simulated ES capacity to ensure queries happen post-overflow.")
    parser.add_argument("--inject_contradiction", action='store_true',
                        help="If set, injects one incorrect fact early and corrects it later.")
    args = parser.parse_args()

    stream = generate_stream(
        num_total_events=args.num_events,
        num_facts=args.num_facts,
        es_capacity=args.es_capacity,
        inject_contradiction=args.inject_contradiction
    )

    # The stream is one long session. We can optionally chunk it into smaller sessions.
    # For now, writing it as one giant session to a single line.
    # A better approach might be to split it for the model. Let's make it one session per file.
    with open(args.output_file, 'w') as f:
        session = {"session_id": "pressure_test_001", "events": stream}
        f.write(json.dumps(session) + '\n')

    print(f"Successfully generated a stream of {len(stream)} events to {args.output_file}")
    print(f"Configuration: {args.num_events} total events, {args.num_facts} facts, ES capacity {args.es_capacity}.")
    if args.inject_contradiction:
        print("Contradictory fact was injected.")


if __name__ == "__main__":
    main()