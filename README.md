# 🤖 TalentScout AI - Intelligent Hiring Assistant

Ever been in a technical interview that felt like being interrogated by a robot? Yeah, me too. That's why I built TalentScout AI - a hiring assistant that actually feels human, while secretly being powered by some pretty cool AI magic.

## 🌟 Live Demo
*[Coming soon - currently running on my local machine eating all my RAM 😅]*

## 🚀 Features

### The Basics (What They Asked For) ✅
- **Conversational Information Gathering** - No more boring forms! Just chat like you're texting a friendly recruiter.
- **Adaptive Question Generation** - The bot's smart enough to know a junior dev shouldn't get grilled about microservices architecture (we've all been there, right?)
- **Real-time Answer Evaluation** - Get instant feedback that's actually helpful, not just "Your response has been recorded" nonsense
- **Comprehensive HR Reports** - Gives recruiters a full picture without making them read through pages of chat logs

### The Fun Stuff I Added Because Why Not 🎯
- **AI-Generated Text Detection** - Catches those sneaky ChatGPT copy-pastes (yes, we can tell when you're cheating 👀)
- **Sentiment Analysis** - If you're stressed, the bot notices and basically gives you a virtual pat on the back
- **Visual Tech Stack Display** - Your skills shown in a cool radar chart because everyone loves good data viz
- **Progress Tracking** - Like a loading bar for interviews - because anxiety loves uncertainty, and we're fixing that
- **Graceful Error Handling** - If something breaks, your answers are safe. I learned this the hard way during testing...
- **Multi-format Reports** - JSON for the systems, pretty text for the humans. Everyone's happy!

## 📸 What It Looks Like

*[Screenshots coming after I figure out how to make my localhost public... or just trust me, it looks awesome]*

### The Chat Interface
Imagine Streamlit, but with custom CSS that doesn't hurt your eyes. Clean blues, friendly grays, and buttons that actually look clickable.

### The Progress Bar
It moves! It updates! It gives you hope that this interview will actually end!

### The Final Report
Charts, scores, insights - basically everything a recruiter needs to make a decision without stalking your LinkedIn.

## 🛠️ Tech Stack (The Nerdy Details)

- **Frontend**: Streamlit (because who has time to build a full React app for a 48-hour assignment?)
- **LLM**: Llama 2 running locally (RIP my laptop's CPU, but hey, no API costs!)
- **AI Detection**: RoBERTa-based classifier (the ChatGPT police 🚔)
- **Visualization**: Plotly (makes everything look 10x more professional)
- **Sentiment Analysis**: TextBlob (simple but effective, like a good coffee)
- **Database**: Your browser's session state (keeping it simple)

## 🏃‍♂️ Want to Run It Yourself?

Fair warning: This thing is hungry for resources. Make sure your laptop is plugged in!

1. Grab the code:
```bash
git clone https://github.com/anmolv11n/talentscout-ai-hiring-assistant.git
cd talentscout-ai-hiring-assistant
```

2. Set up your environment (virtual env recommended unless you like chaos):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the requirements (grab a coffee, this takes a minute):
```bash
pip install -r requirements.txt
```

4. Fire it up:
```bash
streamlit run TalentScoutAI.py
```

5. Open your browser to `http://localhost:8501` and start chatting!

## 💡 Why I Built It This Way

**The Adaptive Questions Thing**: Because asking a bootcamp grad about distributed systems architecture is just mean. The bot adjusts difficulty based on experience - juniors get fundamental questions, seniors get the tough stuff.

**The Cheating Detector**: Look, we all know people paste ChatGPT answers. But instead of blocking it, I just quietly flag it for recruiters. Sneaky? Maybe. Effective? Definitely.

**The Sentiment Analysis**: Technical interviews are stressful enough. If someone's struggling, the bot offers encouragement. It's like having a supportive interviewer who wants you to succeed.

**Local LLM Instead of APIs**: 
- No one's data leaves their computer (privacy win!)
- No API bills (budget win!)
- No rate limits (patience win!)
- My laptop sounds like a jet engine (not really a win...)

## 🎯 The Real Magic

This isn't just a chatbot that asks questions. It's a full interviewing experience that:
- Respects candidates' time with progress tracking
- Gives useful feedback instead of generic responses
- Catches cheaters without being obvious about it
- Actually helps recruiters make better decisions
- Doesn't make people feel like they're talking to a wall

## 🐛 Known "Features" (aka Bugs I'm Calling Features)

- The LLM sometimes takes a coffee break mid-evaluation (that's why there's a timeout)
- Sometimes the sentiment analysis thinks sarcasm is happiness (working on its sense of humor)

## 🔮 If I Had More Time...

- **Voice interviews** - for when typing gets tiring
- **Live coding challenges** - because FizzBuzz never gets old
- **Multiple languages** - ¿Por qué no?
- **Calendar integration** - auto-schedule the next round
- **A "roast my resume" mode** - for the brave souls

## 🤝 Contributing

Found a bug? Want to add a feature? Think my code could be better? You're probably right! Feel free to:
- Open an issue (be nice, I'm sensitive)
- Submit a PR (comments in your code appreciated)
- Fork it and make it better (just give credit, karma's real)

## 👨‍💻 About Me

Just a developer who's been on both sides of terrible technical interviews and thought, "There's gotta be a better way!"

Connect with me:
- LinkedIn: /anmolvohra
- Email: work.anmolvohra@gmail.com
- GitHub: You're already here!

## 📝 License

MIT License - Take it, use it, make it better, sell it to Google, I don't mind! (But maybe buy me a coffee if you get rich?)

## 🙏 Acknowledgments

- My laptop's cooling fan for not giving up
- Stack Overflow for... everything
- The Python community for making cool libraries
- Coffee for making this possible
- You for reading this far!

---

*Built with ❤️, caffeine, and a strong belief that technical interviews don't have to suck*

P.S. If you're a recruiter using this to screen me - yes, I see the irony. Yes, I think it's funny too. 😄
