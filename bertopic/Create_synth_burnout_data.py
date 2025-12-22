# -*- coding: utf-8 -*-
"""
Synthetic DTM Corpus Generator
--------------------------------
Generates a longitudinal text dataset with time-evolving themes and burnout-
conditioned sentences. Outputs a timestamped CSV for BERTopic DTM testing.

How to run (Spyder):
  1) Edit CONFIG below (N_PEOPLE, entry range, T_START/T_END, saving options).
  2) Run the script. Choose a save location (or auto-save to OUTPUT_DIR).
  3) The console prints the absolute path and opens the folder.

Important variables (CONFIG):
  SEED              : int, reproducible randomness.
  N_PEOPLE          : total number of people.
  N_ENT_MIN/MAX     : uniform range of entries per person.
  T_START/T_END     : 'YYYY_MM_DD' date strings.
  ASK_WHERE_TO_SAVE : bool, True => Save-As dialog.
  OUTPUT_DIR        : folder path if dialog is off.
  (Themes and sentence pools are defined below; tweak to taste.)

Output CSV format:
  person_id             str  e.g., 'P001'
  entry_index           int  1..K within person
  timestamp             str  ISO 8601, seconds precision
  text                  str  synthesized text
  latent_burnout_level  float  [0, 1]
  true_theme            str  which theme generated the lead sentence
  event_type            str  personal event token (else empty)

Version history:
  v1.0  (2025-10-21)  Initial merge of evolving themes + burnout sentences.
  v1.1  (2025-10-22)  Progress bar (tqdm), robust saving (Qt/Tk), descriptive
                      filenames (seed/N/entries/timestamp), explorer open.
  v1.2 (2025-10-23)   Saves per-participant csv files, added keywords to themes,
                      added additional themes

Dependencies:
  numpy, pandas, tqdm
  (Optional) PyQt5 or tk for Save-As dialog.

Notes:
  - Outputs are designed to be consumed by BERTopic (see BERT.py).
"""

#%% ==========  Dependencies  ==========
import random
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from pathlib import Path # Saving Path
from tqdm import tqdm # Progress bar
#%% ==========  CONFIG  ==========
SEED         = 1459
N_PEOPLE     = 500          # total number of people
N_ENT_MIN    = 12            # min entries per person
N_ENT_MAX    = 125          # max entries per person
T_START      = "2020_01_02" # YYYY_MM_DD
T_END        = "2029_12_29" # YYYY_MM_DD

# --- Saving options ---
ASK_WHERE_TO_SAVE = True                 # **MAY NOT BE WORKING IF TRUE** --  True = show a "Save As..." dialog
OUTPUT_DIR = None                        # save path (e.g. r"C:\Users\seboe\Documents\DTM")  if None then saves to current working directory

# --- Per-person saving config ---
DATA_ROOT = r"C:\Users\seboe\Documents\Professional\Out of Our Minds\Synthetic_Sentiment\data"         # base folder (relative or absolute). We'll create it if missing.
OPEN_FOLDER_AFTER_SAVE = True

# Keep special individuals like your original script
USE_SPECIAL_TRAJECTORIES = True
WORSENING_ID = "P001"
IMPROVING_ID = "P002"

# Theme evolution curves: increase / decrease / hump over the time window
def curve_increase(x):  # 0 -> 1
    return 0.2 + 0.6 * x

def curve_decrease(x):
    return 0.6 - 0.4 * x

def curve_hump(x):
    # Gaussian bump centered at mid-year
    return 0.2 + 0.6 * np.exp(-((x - 0.5) ** 2) / 0.04)

# REPRODUCIBILITY
random.seed(SEED)
np.random.seed(SEED)

#%% ==========  UTILITIES  ==========
def build_run_id(seed: int, n_people: int, n_ent_min: int, n_ent_max: int,
                 prefix: str = "RUN") -> str:
    """Create a descriptive, timestamped run folder name (no file extension)."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_seed{seed}_N{n_people}_entries{n_ent_min}-{n_ent_max}_{ts}"

def build_default_basename(seed: int, n_people: int, n_ent_min: int, n_ent_max: int,
                           prefix: str = "synthetic_dtm_corpus") -> str:
    """Create a descriptive, timestamped CSV filename."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_seed{seed}_N{n_people}_entries{n_ent_min}-{n_ent_max}_{ts}.csv"

def choose_save_path(default_name="synthetic_dtm_corpus.csv", initial_dir=None):
    """Return a Path chosen via a Save-As dialog.
    Tries Qt (Spyder-friendly), then Tk (topmost), else returns None.
    """
    # --- 1) Try Qt dialog (best inside Spyder) ---
    try:
        from PyQt5 import QtWidgets
        app = QtWidgets.QApplication.instance()
        owns_app = False
        if app is None:
            app = QtWidgets.QApplication([])  # create a temp app if none exists
            owns_app = True

        dlg = QtWidgets.QFileDialog()
        dlg.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        dlg.setNameFilter("CSV files (*.csv)")
        dlg.setDefaultSuffix("csv")
        if initial_dir:
            dlg.setDirectory(str(initial_dir))
        if default_name:
            dlg.selectFile(default_name)
        dlg.setWindowTitle("Save synthetic CSV as...")

        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            path = dlg.selectedFiles()[0]
        else:
            path = None

        if owns_app:
            app.quit()

        return Path(path) if path else None
    except Exception:
        pass

    # --- 2) Fallback to Tk dialog (force on top) ---
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        try:
            root.attributes("-topmost", True)  # make dialog appear on top
        except Exception:
            pass
        root.update()
        path = filedialog.asksaveasfilename(
            title="Save synthetic CSV as...",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialdir=str(initial_dir) if initial_dir else None,
            initialfile=default_name
        )
        try:
            root.destroy()
        except Exception:
            pass
        return Path(path) if path else None
    except Exception:
        return None


def open_folder_in_explorer(folder: Path):
    """Open the folder in the OS file explorer (best-effort)."""
    try:
        import os, sys, subprocess
        if sys.platform.startswith("win"):
            os.startfile(folder)
        elif sys.platform == "darwin":
            subprocess.run(["open", str(folder)], check=False)
        else:
            subprocess.run(["xdg-open", str(folder)], check=False)
    except Exception:
        pass


def parse_date_yyyymmdd(s: str) -> datetime:
    """Parse 'YYYY_MM_DD' into a datetime."""
    try:
        return datetime.strptime(s, "%Y_%m_%d")
    except ValueError as e:
        raise ValueError(f"T_START/T_END must be in 'YYYY_MM_DD' format. Got: {s}") from e

def sample_timestamp_uniform(start_dt: datetime, end_dt: datetime) -> datetime:
    """Sample a timestamp uniformly between start_dt and end_dt (inclusive)."""
    delta = end_dt - start_dt
    seconds = delta.total_seconds()
    if seconds < 0:
        raise ValueError("T_END is earlier than T_START.")
    r = random.random() * seconds
    return start_dt + timedelta(seconds=r)

def frac_between(start_dt: datetime, end_dt: datetime, t: datetime) -> float:
    """Return fraction in [0,1] locating t between start and end."""
    total = (end_dt - start_dt).total_seconds()
    if total <= 0:
        return 0.0
    return np.clip((t - start_dt).total_seconds() / total, 0.0, 1.0)

#%% ==========  THEME DEFINITIONS  ==========
@dataclass
class Theme:
    base: List[str]
    drift: List[str]
    templates: List[str]

THEMES: Dict[str, Theme] = {
    "workload_burnout": Theme(
        base=[
            "burnout","exhausted","overtime","deadline","fatigue","stress","workload","pressure",
            "backlog","overwork","strain","depletion","frustration","overwhelm","weariness","latenight",
            "crunchtime","timepressure","mentaldrain","overcapacity","spiral"
        ],
        drift=[
            "firefighting","scopecreep","contextswtiching","micromanagement","interruptions","meetings",
            "pagerduty","multitasking","triage","saturation","bottleneck","capacitylimit","flareup",
            "deliverable","sprint","escalation","loadspike","rushjob","incident","oncall","hotfix","outage"
        ],
        templates=[
            "I feel {w1} after constant {w2} and {w3}.",
            "The {w2} and {w3} make me {w1}.",
            "Another {w2} means more {w3} and {w1}."
        ],
    ),
    "recovery_holidays": Theme(
        base=[
            "vacation","rest","weekend","holiday","unplug","recover","break","downtime","reset","leisure",
            "recreation","getaway","timeoff","respite","recharge","relaxation","sabbatical","retreat","nap","pause"
        ],
        drift=[
            "beach","hiking","camping","roadtrip","staycation","spaday","meditation","yoga","mindfulness","picnic",
            "backyard","lake","cabin","sightseeing","museum","festival","familyvisit","barbecue","garden","sunrise"
        ],
        templates=[
            "I plan a {w2} to finally {w1}.",
            "The {w2} helped me {w1} last week.",
            "Time to {w1} and fully {w3}."
        ],
    ),
    "management_conflict": Theme(
        base=[
            "manager","meeting","policy","miscommunication","conflict","feedback","alignment","priority","directive",
            "decision","restructure","accountability","expectations","performance","escalation","approval","stakeholder",
            "roadmap","resourcing","deadlinechange"
        ],
        drift=[
            "reorg","orgchart","OKR","KPI","townhall","memo","offsite","statusupdate","churn","attrition","headcount",
            "hiringfreeze","pivot","coursecorrection","audit","compliance","governance","mediation","retro","standup"
        ],
        templates=[
            "There was {w1} about a new {w2} in the {w3}.",
            "Our {w1} scheduled a {w2} after the {w3}.",
            "The {w1} caused {w2} and more {w3}."
        ],
    ),
    "personal_events": Theme(
        base=[
            "promotion","hired","onboarding","fired","laidoff","demoted","relationshipstarted","breakup","wedding",
            "engagement","newbaby","bereavement","relocation","moved","graduation","anniversary","retirement",
            "illnessinfamily","caregiverrole","homepurchase"
        ],
        drift=[
            "careerchange","sabbatical","familyemergency","visaupdate","adoption","separation","remarriage","housesale",
            "petadoption","travelmove","accident","surgery","legalmatter","mentorship","apprenticeship","internship",
            "reunion","milestone","award","setback"
        ],
        templates=[
            "Personal update: {w1} and dealing with {w2} while planning for {w3}.",
            "Life event: {w1}. It affected {w2} and led to {w3}.",
            "At home, {w1}; now navigating {w2} and {w3}."
        ],
    ),
    "emotional_state": Theme(
        base=[
            "anxious","depressed","happy","fulfilled","included","excluded","calm","tense","overwhelmed","relieved",
            "grateful","irritable","lonely","motivated","hopeful","discouraged","confident","uncertain","frustrated","serene"
        ],
        drift=[
            "content","burnedout","stuck","energized","apathetic","jittery","balanced","optimistic","pessimistic",
            "wistful","elated","numb","reflective","distracted","mindful","restless","centered","moody","buoyant","flat"
        ],
        templates=[
            "Lately I feel {w1}, sometimes {w2}, and occasionally {w3}.",
            "My mood has been {w1}, moving between {w2} and {w3}.",
            "Emotionally I am {w1}; it shifts toward {w2} with bouts of {w3}."
        ],
    ),
    "work_changes": Theme(
        base=[
            "deadline","promotion","reprimand","review","raise","probation","training","support","backlog","handover",
            "transfer","reassignment","bonus","appraisal","mentorship","crosstrain","certification","onboarding",
            "offboarding","policyupdate"
        ],
        drift=[
            "OKR","KPI","audit","performanceplan","escalation","milestone","sprint","roadmapchange","bugbash","incident",
            "postmortem","rollout","pilot","beta","launch","patch","refactor","deprecation","migration","rescope"
        ],
        templates=[
            "Work update: {w1} with a {w2} after the {w3}.",
            "I have a {w1} and got a {w2} following a {w3}.",
            "We hit a {w1}, received {w2}, and started {w3}."
        ],
    ),
    "medical_wellbeing": Theme(
        base=[
            "appointment","checkup","diagnosis","treatment","medication","therapy","physical","labresults","vaccination",
            "followup","referral","clinic","insurance","symptom","recovery","prognosis","monitoring","sideeffects",
            "telehealth","screening"
        ],
        drift=[
            "surgery","imaging","MRI","CTscan","bloodwork","biopsy","rehab","physiotherapy","specialist","anesthesia",
            "discharge","inpatient","outpatient","prescription","dosage","adherence","compliance","flareup","remission",
            "complication","outcome"
        ],
        templates=[
            "Health update: {w1} and {w2} with notes about {w3}.",
            "The {w1} led to {w2}; monitoring {w3}.",
            "Discussed {w1} and started {w2} to address {w3}."
        ],
    ),
    "physiological_state": Theme(
        base=[
            "sick","nauseous","headache","stomachache","cramps","fatigue","sore","dizzy","feverish","chills",
            "congestion","cough","sorethroat","bloated","jetlagged","restless","energized","strong","limber","refreshed"
        ],
        drift=[
            "foodpoisoning","stomachbug","dehydration","DOMS","musclestrain","sprain","migraine","allergies","heartburn",
            "indigestion","insomnia","overslept","underslept","crampssubsided","energysurge","runnershigh","endorphins",
            "tighthamstrings","lacticacid","appetiteloss"
        ],
        templates=[
            "Physically I feel {w1}; earlier it was {w2} with some {w3}.",
            "Body check: {w1} today, dealing with {w2} and {w3}.",
            "Noticed {w1} and episodes of {w2} alongside {w3}."
        ],
    ),
    "social_events": Theme(
        base=[
            "dinner","lunch","coffee","brunch","hangout","party","gathering","meetup","gamenight","movienight",
            "celebration","reunion","catchup","outing","conversation","invite","host","guest","friendgroup","colleague"
        ],
        drift=[
            "newacquaintance","networking","communityevent","conflict","disagreement","apology","reconciliation",
            "planchange","raincheck","surprisevisit","doubledate","potluck","festival","concert","sportsgame",
            "volunteering","bookclub","babyshower","housewarming","farewell"
        ],
        templates=[
            "Had a {w1} with friends and a {w2}; it led to {w3}.",
            "Social plan: {w1}, then {w2}, lots of {w3}.",
            "We organized a {w1}; there was {w2} and later {w3}."
        ],
    ),
}
# --- Add new themes ---
THEMES.update({
    "financial_stress_budgeting": Theme(
        base=[
            "bills","budget","savings","debt","expenses","paycheck","rent","mortgage","interest","inflation",
            "credit","loan","fees","insurance","taxes","emergencyfund","balance","overdraft","cashflow","allowance",
            "deduction","withholding","surpriseexpense","latefee","expensecap"
        ],
        drift=[
            "sidehustle","investment","stocks","bonds","market","recession","bonus","raise","cutbacks","subscription",
            "cancellation","refinance","consolidation","delinquency","paymentplan","installment","windfall","grant",
            "stipend","dividend","crypto","brokerage","budgetapp","spendingfreeze","cashenvelope"
        ],
        templates=[
            "Money update: {w1} and {w2} while tracking {w3}.",
            "We reviewed {w1}, adjusted for {w2}, and watched {w3}.",
            "Facing {w1} and planning around {w2} with {w3}."
        ],
    ),
    "commute_transportation": Theme(
        base=[
            "commute","traffic","transit","subway","bus","parking","carpool","ride","delay","detour",
            "construction","accident","bridge","ferry","timetable","schedule","ticket","platform","congestion","rushhour",
            "laneclosure","detainment","speedlimit","tollbooth","pedestrian"
        ],
        drift=[
            "rideshare","scooter","bike","flattire","breakdown","tow","maintenance","charging","evstation","reroute",
            "snowplow","blackice","roadwork","signalfailure","shuttle","aisle","seat","crowding","closure","diversion",
            "stormwarning","trackwork","signalupgrade","lanechange","detector"
        ],
        templates=[
            "Commute note: {w1} with {w2} and a bit of {w3}.",
            "Caught {w1}; then {w2}; finally arrived after {w3}.",
            "We had {w1} and {w2}; the {w3} slowed everything."
        ],
    ),
    "caregiving_parenting": Theme(
        base=[
            "childcare","daycare","school","homework","pickup","dropoff","bedtime","diapers","feeding","pediatrician",
            "playdate","tantrum","milestone","pottytraining","parentteacher","babysitter","nanny","recital","sportspractice","fieldtrip",
            "storytime","lunchbox","permissionform","carline","aftercare"
        ],
        drift=[
            "sickday","fever","rash","immunization","lice","snowday","parentvolunteer","fundraiser","carseat","stroller",
            "playground","screentime","bedtimebattle","growthspurt","sleeptrain","reportcard","sciencefair","audition","summercamp","collegevisit",
            "orientation","homeschool","virtualclass","extracurricular","enrichment"
        ],
        templates=[
            "Parenting: {w1} plus {w2}, and we handled {w3}.",
            "Today involved {w1}; later {w2}; some {w3} as well.",
            "We coordinated {w1} and addressed {w2} amid {w3}."
        ],
    ),
    "housing_home_projects": Theme(
        base=[
            "rent","lease","landlord","neighbor","utility","water","power","internet","repair","maintenance",
            "appliance","plumbing","electrical","painting","renovation","mortgage","inspection","pestcontrol","roofing","yardwork",
            "trash","recycling","security","gate","mailbox"
        ],
        drift=[
            "landscaping","gardening","compost","declutter","moveout","movein","storage","delivery","contractor","permit",
            "zoning","noisecomplaint","securitydeposit","thermostat","insulation","flooding","mold","leak","handyman","foundation",
            "drywall","tiling","countertop","cabinetry","weatherproofing"
        ],
        templates=[
            "Home update: {w1} and {w2} with notes about {w3}.",
            "Worked on {w1}, scheduled {w2}, and fixed {w3}.",
            "We planned {w1}; contacted {w2}; resolved {w3}."
        ],
    ),
    "hobbies_creativity": Theme(
        base=[
            "music","guitar","piano","singing","writing","reading","poetry","painting","drawing","sketch",
            "photography","cooking","baking","knitting","crochet","pottery","gardening","gaming","hiking","journaling",
            "calligraphy","watercolor","oilpaint","sculpture","origami"
        ],
        drift=[
            "podcast","editing","streaming","coding","robotics","printing3d","cosplay","woodworking","metalwork","boardgames",
            "tabletop","chess","marathon","triathlon","scrapbooking","beadwork","filmmaking","animation","mixing","sampling",
            "djset","sounddesign","blogging","zine","letterpress"
        ],
        templates=[
            "For fun: {w1} and {w2}, experimenting with {w3}.",
            "I practiced {w1}, tried {w2}, and explored {w3}.",
            "Creative time included {w1} plus {w2} and {w3}."
        ],
    ),
    "learning_profdev": Theme(
        base=[
            "course","class","lecture","seminar","workshop","certification","training","syllabus","exam","project",
            "readinglist","mentor","mentee","internship","residency","conference","poster","paper","thesis","capstone",
            "portfolio","officehours","studygroup","notes","quiz"
        ],
        drift=[
            "webinar","tutorial","online","assignment","labwork","datasci","statistics","model","prototype","grantwriting",
            "peerreview","publication","networking","keynote","breakout","hackathon","fellowship","scholarship","bootcamp","summerinstitute",
            "literaturereview","preprint","slides","recording","demo"
        ],
        templates=[
            "Learning: {w1} and {w2} while preparing {w3}.",
            "I joined {w1}, submitted {w2}, and iterated on {w3}.",
            "We studied {w1}, practiced {w2}, and presented {w3}."
        ],
    ),
    "nutrition_diet": Theme(
        base=[
            "breakfast","lunch","dinner","snack","mealprep","groceries","produce","protein","carbs","fats",
            "fiber","hydration","vitamins","portion","craving","appetite","caffeine","tea","coffee","dessert",
            "salad","sauce","seasoning","leftovers","pantry"
        ],
        drift=[
            "fasting","keto","vegan","vegetarian","glutenfree","dairyfree","intolerance","allergy","cheatmeal","sugarcrash",
            "bloating","indigestion","heartburn","probiotic","electrolyte","supplement","smoothie","juicing","mealplan","macrotracking",
            "batchcook","airfryer","sheetpan","fermented","fiberboost"
        ],
        templates=[
            "Food log: {w1} and {w2}, then {w3}.",
            "I prepped {w1}, cut back on {w2}, added {w3}.",
            "Meals were {w1}; cravings for {w2}; tracked {w3}."
        ],
    ),
    "sleep_recovery": Theme(
        base=[
            "bedtime","wakeup","alarm","nap","insomnia","oversleep","undersleep","dream","nightmare","jetlag",
            "routine","blackoutcurtains","earplugs","melatonin","restfulness","sleepquality","light","noise","snoring","winddown",
            "restday","sleepwindow","siesta","catnap","slumber"
        ],
        drift=[
            "deepsleep","remsleep","sleepcycle","nightsweats","nightshift","shiftwork","sleeptracker","bluelight","caffeinecutoff","bedtimealarm",
            "latenight","earlyrise","dawn","evening","circadian","sleepdebt","sleepbank","powernap","sleephygiene","sleeprestriction",
            "chronotype","jetlagrecovery","earlieroutine","winddownmusic","weightedblanket"
        ],
        templates=[
            "Sleep update: {w1} and {w2} with notes on {w3}.",
            "Last night had {w1}; today I felt {w2}; tracking {w3}.",
            "Working on {w1}, avoiding {w2}, aiming for {w3}."
        ],
    ),
    "technology_tools": Theme(
        base=[
            "laptop","desktop","server","network","wifi","hotspot","router","firewall","software","update",
            "version","bug","ticket","issue","feature","workflow","script","automation","dashboard","repository",
            "licensing","endpoint","monitoring","alert","database"
        ],
        drift=[
            "outage","downtime","incident","rollback","deploy","release","patch","hotfix","sandbox","staging",
            "production","credential","access","permission","vpn","encryption","backup","restore","container","notebook",
            "pipeline","orchestration","microservice","logrotate","observability"
        ],
        templates=[
            "Tech note: {w1} with {w2} and a {w3}.",
            "We handled {w1}, then {w2}, followed by {w3}.",
            "Set up {w1}; encountered {w2}; shipped {w3}."
        ],
    ),
    "weather_environment": Theme(
        base=[
            "weather","rain","storm","snow","heat","cold","wind","humidity","sunshine","cloud",
            "fog","frost","pollen","airquality","wildfire","flood","drought","ice","thunder","lightning",
            "breeze","overcast","drizzle","mist","hail"
        ],
        drift=[
            "heatwave","coldsnap","hailstorm","blizzard","smog","haze","uvindex","sunrise","sunset","seasonal",
            "daylength","eclipse","rainbow","mudslide","landslide","duststorm","sandstorm","monsoon","thaw","freeze",
            "stormsurge","microburst","downpour","chinook","whiteout"
        ],
        templates=[
            "Weather: {w1} with {w2}; later saw {w3}.",
            "Forecast showed {w1}; we had {w2}; brief {w3}.",
            "Conditions were {w1}; alerts for {w2}; some {w3}."
        ],
    ),
})

# --- Social wellbeing and perception-of-self themes (NEW) ---
THEMES.update({
    "social_self_esteem": Theme(
        base=[
            "confident","capable","respected","accepted","valued","secure","worthwhile","adequate","selfassured","satisfied",
            "optimistic","competent","selfreliant","balanced","grounded","resilient","selftrust","proud","steady","hopeful",
            "positive","assured","appreciated","supported","included"
        ],
        drift=[
            "unpopular","worthless","doubt","selfcriticism","insecure","ashamed","embarrassed","isolated","excluded","inferior",
            "uncertain","awkward","selfdoubt","fragile","uneasy","timid","selfconscious","regret","rumination","disheartened",
            "discouraged","vulnerable","wavering","imposter","anxious"
        ],
        templates=[
            "I feel {w1} about myself; sometimes {w2}; overall aiming to stay {w3}.",
            "Lately my selfview is {w1}, with moments of {w2}, moving toward {w3}.",
            "My selfesteem feels {w1}; it drifts toward {w2} and back to {w3}."
        ],
    ),
    "social_boldness": Theme(
        base=[
            "assertive","outspoken","spokesperson","firstmove","initiative","volunteer","lead","present","pitch","negotiate",
            "challenge","debate","moderate","facilitate","host","organize","introduce","network","icebreaker","proactive",
            "visible","vocal","decisive","frontfoot","selfstarter"
        ],
        drift=[
            "reserved","hesitant","observer","quiet","reticent","withdrawn","passive","backseat","lowkey","softspoken",
            "cautious","unsure","tentative","nervous","avoidant","sidelines","understated","reluctant","muted","shy",
            "deferential","shrinking","timorous","wavering","unassertive"
        ],
        templates=[
            "In groups I tend to be {w1}; at times I become {w2}, though I aim to stay {w3}.",
            "Socially I act {w1}, occasionally {w2}, trying to remain {w3}.",
            "During meetings I am {w1}; sometimes {w2}; working toward {w3}."
        ],
    ),
    "sociability": Theme(
        base=[
            "sociable","gregarious","mingling","networking","chatty","outgoing","friendly","warm","approachable","companionship",
            "community","belonging","teamwork","collaboration","smalltalk","meetups","gathering","club","cozy","welcoming",
            "companions","togetherness","bonding","affiliation","connectedness"
        ],
        drift=[
            "solitude","withdrawal","aloof","detached","reserved","asocial","isolation","antisocial","loner","avoidance",
            "distant","coolness","standoffish","quietude","seclusion","homebound","introverted","lowcontact","limitedreach","closedoff",
            "reclusive","independent","selfcontained","unsocial","separate"
        ],
        templates=[
            "I feel {w1} and enjoy {w2}; when tired I lean toward {w3}.",
            "My social energy is {w1}; I like {w2}, though I drift into {w3}.",
            "Day to day I am {w1}; I seek {w2}; sometimes I prefer {w3}."
        ],
    ),
    "liveliness": Theme(
        base=[
            "cheerful","optimistic","dynamic","lively","energetic","upbeat","spark","vibrant","animated","spirited",
            "bright","bubbly","peppy","jaunty","spry","buoyant","sunny","positive","highenergy","zestful",
            "enthused","motivated","eager","engaged","lighthearted"
        ],
        drift=[
            "sluggish","subdued","lethargic","drained","flat","downcast","weary","lowenergy","glum","gloomy",
            "apathetic","blunted","dull","plodding","heavy","spiritless","tired","washedout","burned","listless",
            "dreary","blue","inactive","unmotivated","foggy"
        ],
        templates=[
            "Most days I feel {w1}; sometimes I slip into {w2} before returning to {w3}.",
            "My mood is {w1}, with periods of {w2}; I try to keep things {w3}.",
            "Energy feels {w1}; occasionally {w2}; aiming for {w3}."
        ],
    ),
})


# Burnout-flavored sentences
LOW_POOL = [
    "Feeling focused and energized today.",
    "Workload feels manageable and I made good progress.",
    "Team support is strong and communication is clear.",
    "I took breaks and my sleep was solid last night.",
    "Motivated about current projects and goals.",
    "I feel balanced and productive.",
    "Today felt efficient with minimal distractions.",
    "Confidence is high and tasks are clear.",
]
MID_POOL = [
    "Some tasks are dragging and I'm a bit distracted.",
    "Progress is okay but I'm juggling competing priorities.",
    "Energy is uneven and I'm working through minor setbacks.",
    "Pressure is noticeable but still under control.",
    "Sleep wasn't perfect and focus came in waves.",
    "Deadlines are stacking up; I'm pacing myself.",
    "I'm coping but it's getting hectic.",
]
HIGH_POOL = [
    "Exhausted and overwhelmed by deadlines.",
    "Struggling to concentrate; tasks feel impossible.",
    "Feeling cynical and detached from work.",
    "Back-to-back demands with no recovery time.",
    "Mentally drained; motivation is collapsing.",
    "Stressed, irritable, and falling behind.",
    "Burnout is spiking; I can't keep up.",
    "Workload is crushing and sleep is awful.",
]
EXTRAS_LOW  = ["Had a short walk.", "Took time to plan.", "Hydrated well.", "Met a small milestone."]
EXTRAS_HIGH = ["Headaches persisted.", "Skipped lunch to meet a deadline.", "Woke up multiple times.", "Emails piled up."]

def burnout_sentences(level: float) -> str:
    """Generate 1–3 sentences conditioned on burnout level in [0,1]."""
    # weights for low/mid/high pools
    p_low  = max(0.0, 1.0 - 1.5 * level)
    p_high = max(0.0, 1.5 * level - 0.3)
    p_mid  = max(0.0, 1.0 - p_low - p_high)
    probs  = np.array([p_low, p_mid, p_high])
    probs  = probs / probs.sum() if probs.sum() > 0 else np.array([1/3, 1/3, 1/3])

    pools = [LOW_POOL, MID_POOL, HIGH_POOL]
    n_sent = random.choice([1, 2])  # keep these short; main content comes from theme
    out = []
    for _ in range(n_sent):
        pool_idx = np.random.choice([0, 1, 2], p=probs)
        out.append(random.choice(pools[pool_idx]))
    if level < 0.35 and random.random() < 0.4:
        out.append(random.choice(EXTRAS_LOW))
    if level > 0.65 and random.random() < 0.4:
        out.append(random.choice(EXTRAS_HIGH))
    return " ".join(out)

def sample_theme_sentence(theme: Theme, t_frac: float, drift_strength: float = 0.6) -> Tuple[str, None]:
    """Pick a template and fill with base/drift words (more drift later in the timeline)."""
    prob_drift = drift_strength * np.clip(t_frac, 0.0, 1.0)
    def pick_word():
        if theme.drift and np.random.rand() < prob_drift:
            return random.choice(theme.drift)
        return random.choice(theme.base)
    tpl = random.choice(theme.templates)
    return tpl.format(w1=pick_word(), w2=pick_word(), w3=pick_word()), None

def sample_personal_event_sentence(theme: Theme, t_frac: float) -> Tuple[str, str]:
    """Special case to record an event_type when theme is personal_events."""
    sentence, _ = sample_theme_sentence(theme, t_frac, drift_strength=0.8)
    # Extract a likely 'event' token from base+drift to store in event_type
    event_type = random.choice(theme.base + theme.drift)
    return sentence, event_type

def theme_probabilities(t_frac: float, burnout_level: float, theme_names: List[str]) -> np.ndarray:
    """
    Build a probability vector over themes that changes over time AND
    responds to current burnout level. Robust to theme additions.
    """
    # Time shapes
    hump     = np.exp(-((t_frac - 0.5) ** 2) / 0.04)     # mid-window bump
    sin_fast = np.sin(12 * np.pi * t_frac)               # faster oscillation
    sin_med  = np.sin(8 * np.pi * t_frac)
    sin_year = np.sin(2 * np.pi * t_frac)
    sin_sq   = sin_year ** 2

    # Core + previously added themes (unchanged where sensible)
    known = {
        "workload_burnout":        0.10 + 0.60 * t_frac + 0.50 * burnout_level,
        "recovery_holidays":       0.10 + 0.50 * (1.0 - t_frac) + 0.20 * (1.0 - burnout_level),
        "management_conflict":     0.10 + 0.60 * hump,
        "personal_events":         0.08 + 0.06 * np.sin(6 * np.pi * t_frac),
        "emotional_state":         0.15 + 0.50 * abs(burnout_level - 0.5),
        "work_changes":            0.10 + 0.10 * np.random.rand(),
        "medical_wellbeing":       0.08 + 0.30 * burnout_level + 0.05 * np.random.rand(),
        "physiological_state":     0.12 + 0.40 * abs(burnout_level - 0.5) + 0.08 * np.random.rand(),
        "social_events":           0.12 + 0.10 * sin_med + 0.05 * np.random.rand(),
        "financial_stress_budgeting": 0.10 + 0.45 * burnout_level + 0.10 * sin_fast,
        "commute_transportation":     0.12 + 0.20 * np.abs(np.sin(10 * np.pi * t_frac)) + 0.05 * np.random.rand(),
        "caregiving_parenting":       0.10 + 0.30 * (np.sin(4 * np.pi * t_frac) ** 2),
        "housing_home_projects":      0.10 + 0.40 * np.exp(-((t_frac - 0.4) ** 2) / 0.03),
        "hobbies_creativity":         0.12 + 0.40 * (1.0 - burnout_level),
        "learning_profdev":           0.10 + 0.20 * (np.sin(4 * np.pi * t_frac) ** 2),
        "nutrition_diet":             0.10 + 0.15 * sin_sq + 0.05 * np.random.rand(),
        "sleep_recovery":             0.14 + 0.50 * abs(burnout_level - 0.5) + 0.06 * np.random.rand(),
        "technology_tools":           0.10 + 0.10 * np.random.rand() + 0.08 * np.abs(sin_fast),
        "weather_environment":        0.10 + 0.30 * sin_sq,

        # NEW social wellbeing themes
        # Self-esteem content tends to surface more when burnout is high or fluctuating
        "social_self_esteem":     0.12 + 0.40 * abs(burnout_level - 0.5) + 0.20 * burnout_level,
        # Boldness shows up more when burnout is lower (energy to initiate), with small oscillation
        "social_boldness":        0.10 + 0.55 * (1.0 - burnout_level) + 0.05 * np.abs(sin_year),
        # Sociability preference rises when burnout is lower; periodic bumps for social cycles
        "sociability":            0.12 + 0.45 * (1.0 - burnout_level) + 0.08 * (sin_med ** 2),
        # Liveliness strongly anticorrelated with burnout; small seasonal shape
        "liveliness":             0.14 + 0.60 * (1.0 - burnout_level) + 0.06 * sin_sq,
    }

    default_prior = 0.05
    w = np.array([max(1e-6, known.get(name, default_prior)) for name in theme_names], dtype=float)
    w = w / w.sum()
    return w



#%% ==========  CORPUS GENERATOR  ==========
def generate_corpus(n_people: int,
                    n_ent_min: int,
                    n_ent_max: int,
                    t_start: str,
                    t_end: str,
                    use_special: bool = True,
                    worsening_id: str = "P001",
                    improving_id: str = "P002",
                    seed: Optional[int] = None) -> pd.DataFrame:
    """Create the synthetic corpus as a DataFrame (with a progress bar)."""
    if seed is not None:
        random.seed(seed); np.random.seed(seed)

    start_dt = parse_date_yyyymmdd(t_start)
    end_dt   = parse_date_yyyymmdd(t_end)

    people = [f"P{str(i+1).zfill(3)}" for i in range(n_people)]

    # ---- Pre-sample entries per person so we know the total rows (for tqdm total) ----
    entries_per_person = {pid: int(np.random.randint(n_ent_min, n_ent_max + 1)) for pid in people}
    total_rows = int(sum(entries_per_person.values()))

    rows = []

    # One progress bar that advances for every entry we generate
    pbar = tqdm(total=total_rows, desc="Generating entries",
            unit="entry", dynamic_ncols=True, mininterval=0.2,
            miniters=1, leave=True)

    for pid in people:
        # Person-specific trajectory (same logic you had)
        if use_special and pid == worsening_id:
            baseline = 0.20
            trend    = +0.50   # across the whole window (0→1)
        elif use_special and pid == improving_id:
            baseline = 0.80
            trend    = -0.50
        else:
            baseline = float(np.clip(np.random.beta(2, 3), 0.05, 0.95))  # skew low
            trend    = float(np.random.normal(0.0, 0.15))                # gentle drift

        volatility = float(abs(np.random.normal(0.08, 0.03)))            # per-entry noise

        # Use the pre-sampled entry count
        n_entries = entries_per_person[pid]

        # Sample timestamps then sort so entry_index follows time
        stamps = [sample_timestamp_uniform(start_dt, end_dt) for _ in range(n_entries)]
        stamps.sort()

        # (Optional) show which person is being processed
        pbar.set_description(f"Generating entries (person {pid})")

        for idx, ts in enumerate(stamps, start=1):
            t_frac  = frac_between(start_dt, end_dt, ts)
            latent  = float(np.clip(baseline + trend * t_frac + np.random.normal(0.0, volatility), 0.0, 1.0))

            # Pick a theme (time-varying + burnout-aware)
            theme_names = list(THEMES.keys())
            probs = theme_probabilities(t_frac, latent, theme_names)
            theme_idx   = int(np.random.choice(len(theme_names), p=probs))
            theme_name  = theme_names[theme_idx]
            theme_obj   = THEMES[theme_name]

            if theme_name == "personal_events":
                theme_sentence, event_type = sample_personal_event_sentence(theme_obj, t_frac)
            else:
                theme_sentence, event_type = sample_theme_sentence(theme_obj, t_frac)

            burnout_text = burnout_sentences(latent)
            text = f"{theme_sentence} {burnout_text}".strip()

            rows.append({
                "person_id": pid,
                "entry_index": idx,
                "timestamp": ts.isoformat(timespec="seconds"),
                "text": text,
                "latent_burnout_level": round(latent, 3),
                "true_theme": theme_name,
                "event_type": event_type if event_type is not None else ""
            })

            # Advance the bar for every row created
            pbar.update(1)
            if (idx % 200) == 0:   # every ~200 entries
                pbar.refresh()     # force a repaint in Spyder

    df = pd.DataFrame(rows).sort_values(["person_id", "timestamp"]).reset_index(drop=True)
    return df

#%% ==========  RUN  ==========
if __name__ == "__main__":
    # 1) Generate the full cohort DataFrame (unchanged)
    df = generate_corpus(
        n_people=N_PEOPLE,
        n_ent_min=N_ENT_MIN,
        n_ent_max=N_ENT_MAX,
        t_start=T_START,
        t_end=T_END,
        use_special=USE_SPECIAL_TRAJECTORIES,
        worsening_id=WORSENING_ID,
        improving_id=IMPROVING_ID,
        seed=SEED
    )

    # 2) Build RUNID and create Data/RUNID folder
    data_root = Path(DATA_ROOT) if DATA_ROOT else Path.cwd() / "Data"
    run_id = build_run_id(SEED, N_PEOPLE, N_ENT_MIN, N_ENT_MAX)
    run_dir = data_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # 3) Save one CSV PER PERSON (P001.csv, P002.csv, ...)
    n_files = 0
    for pid, dfp in df.groupby("person_id", sort=True):
        out_path = run_dir / f"{pid}.csv"
        dfp.to_csv(out_path, index=False, encoding="utf-8")
        n_files += 1

    # 4) Clear, explicit logging
    print("\n=== Synthetic Per-Person CSVs Saved ===")
    print(" Data root          :", data_root.resolve())
    print(" RUNID folder       :", run_dir.resolve())
    print(" Seed               :", SEED)
    print(" N participants     :", N_PEOPLE)
    print(" Entries/person     :", f"{N_ENT_MIN}–{N_ENT_MAX}")
    print(" Date range         :", f"{T_START} to {T_END}")
    print(" Total rows (cohort):", len(df))
    print(" Files written      :", n_files)
    print(" First few rows (cohort):")
    print(df.head(5).to_string(index=False))
    print("========================================\n")

    if OPEN_FOLDER_AFTER_SAVE:
        open_folder_in_explorer(run_dir)

