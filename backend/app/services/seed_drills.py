"""Seed practice drills into the database."""
from app.models import PracticeDrill

DRILLS = [
    PracticeDrill(
        id=1,
        title="Still Head Drill",
        shot_category=None,
        technique_focus="head_position",
        difficulty="beginner",
        duration_minutes=15,
        description=(
            "Place a sticker or mark on a wall at eye height. Stand in batting stance and "
            "shadow-play shots while keeping your eyes fixed on the mark. The goal is to "
            "maintain a perfectly still head position throughout each swing."
        ),
        steps=[
            "Set up in a batting stance facing a wall mark at eye level.",
            "Without a ball, shadow the full motion of your chosen shot.",
            "Keep your eyes on the wall mark — do not let your head tilt or move laterally.",
            "Repeat 20 times, rest, then repeat with a different shot.",
            "Record yourself from the front to self-review head movement.",
        ],
        equipment=["Cricket bat", "Wall with a mark or sticker"],
        tips=[
            "Imagine your head is balanced on top of a pole — it should never rock side to side.",
            "The Bradman method: if you have a ball, toss it into the air above your head and hit it while watching it.",
        ],
    ),
    PracticeDrill(
        id=2,
        title="High Elbow Drive Drill",
        shot_category="cover_drive",
        technique_focus="front_arm",
        difficulty="beginner",
        duration_minutes=20,
        description=(
            "Practice the cover drive with conscious attention to keeping the front elbow high "
            "and leading the downswing. Use a tee or have a partner feed underarm balls."
        ),
        steps=[
            "Set up a batting tee at just above knee height.",
            "Stand in normal stance, front foot slightly pointed toward cover.",
            "Focus on leading the downswing with the front elbow, not the hands.",
            "Drive through the ball toward cover — hold the follow-through position for 2 seconds.",
            "Repeat 15–20 times. Check your elbow position in a mirror or video.",
        ],
        equipment=["Cricket bat", "Batting tee or underarm feeder", "Ball"],
        tips=[
            "Think 'elbow first' on the downswing.",
            "The bat face should be angled toward cover at the point of contact, not closed.",
        ],
    ),
    PracticeDrill(
        id=3,
        title="Back Arm Isolation Drill",
        shot_category=None,
        technique_focus="back_arm",
        difficulty="beginner",
        duration_minutes=10,
        description=(
            "One-handed drill using only the back (bottom) hand to feel the role the bottom "
            "hand plays in driving power through impact."
        ),
        steps=[
            "Hold the bat with only your bottom hand (the driving hand).",
            "Shadow-play a drive, focusing on pushing through with the bottom hand.",
            "The wrist should roll naturally through impact — do not lock.",
            "After 10 reps, add the top hand back and notice the coordination.",
        ],
        equipment=["Cricket bat"],
        tips=[
            "The bottom hand generates power; the top hand provides direction.",
        ],
    ),
    PracticeDrill(
        id=4,
        title="Wide Stance Balance Drill",
        shot_category=None,
        technique_focus="stance",
        difficulty="beginner",
        duration_minutes=10,
        description=(
            "Find your optimal stance width by experimenting between narrow and wide stances "
            "and observing the effect on balance and mobility."
        ),
        steps=[
            "Stand with feet very close together. Shadow-play a drive and feel how easy it is to move.",
            "Move to shoulder-width stance. Repeat.",
            "Move to wider than shoulder width. Note how mobility decreases.",
            "Find the position where you feel most balanced and can move freely.",
            "Mark that position and use it as your starting stance.",
        ],
        equipment=["Cricket bat", "Tape or chalk to mark foot positions"],
        tips=[
            "The ideal stance allows you to move quickly to both a front-foot and back-foot shot.",
            "Different players use different widths — Virat Kohli's is slightly wider than average.",
        ],
    ),
    PracticeDrill(
        id=5,
        title="One-Leg Balance Drill",
        shot_category=None,
        technique_focus="balance",
        difficulty="beginner",
        duration_minutes=10,
        description=(
            "Improve balance and weight transfer by practicing shots while standing on one leg."
        ),
        steps=[
            "Stand on your front foot only, back foot raised slightly off the ground.",
            "Shadow-play front-foot drives 10 times.",
            "Switch to back foot only, front foot raised.",
            "Shadow-play back-foot shots 10 times.",
            "This builds proprioception and strengthens the stabilising muscles used in batting.",
        ],
        equipment=["Cricket bat"],
        tips=[
            "Keep your head still even as you balance.",
            "If this is too easy, try closing your eyes.",
        ],
    ),
    PracticeDrill(
        id=6,
        title="Cone Follow-Through Drill",
        shot_category=None,
        technique_focus="follow_through",
        difficulty="beginner",
        duration_minutes=15,
        description=(
            "Place cones to guide your bat's follow-through path, ensuring you complete the "
            "stroke and don't decelerate before contact."
        ),
        steps=[
            "Place a cone where you intend the ball to land after the shot.",
            "Set another cone as a bat follow-through guide beyond impact.",
            "Drive or play the shot — the bat must follow through past the second cone.",
            "Stopping early signals poor commitment through the ball.",
        ],
        equipment=["Cricket bat", "2 cones or markers"],
        tips=[
            "Accelerate through impact, not to impact.",
        ],
    ),
    PracticeDrill(
        id=7,
        title="Shadow Footwork Drill",
        shot_category=None,
        technique_focus="footwork",
        difficulty="beginner",
        duration_minutes=15,
        description=(
            "Use a metronome or partner calling 'front' and 'back' to practice rapid footwork "
            "without a ball."
        ),
        steps=[
            "Set a metronome to 60 BPM.",
            "On each beat, alternate between moving to a front-foot and back-foot position.",
            "Focus on landing in a balanced, wide stance each time.",
            "Increase tempo to 80 BPM after 2 minutes.",
        ],
        equipment=["Metronome or phone app", "Cricket bat (optional)"],
        tips=[
            "Quick feet → better shot selection → more runs.",
            "The best batters move their feet before the ball is even halfway down the pitch.",
        ],
    ),
]


def seed_drills(db):
    """Insert drills if they don't already exist."""
    for drill in DRILLS:
        existing = db.query(PracticeDrill).filter(PracticeDrill.id == drill.id).first()
        if not existing:
            db.add(drill)
    db.commit()
