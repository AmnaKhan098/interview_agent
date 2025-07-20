"use client";

import { motion } from "framer-motion";
import { useState, useEffect } from "react";

// Dummy data for behavioral questions
const questions = [
  {
    id: 1,
    question: "Tell me about a time you faced a challenge at work.",
    hint: "Focus on STAR: Situation, Task, Action, Result.",
  },
  {
    id: 2,
    question: "Describe a situation where you showed leadership.",
    hint: "Highlight initiative and impact.",
  },
  {
    id: 3,
    question: "How do you handle tight deadlines?",
    hint: "Mention prioritization and communication.",
  },
];

// Inline QuestionCard component (could be moved to components/ui if reused)
function QuestionCard({ question, hint }: { question: string; hint: string }) {
  return (
    <motion.div
      whileHover={{ scale: 1.03 }}
      whileTap={{ scale: 0.98 }}
      className="bg-white/80 border border-gray-200 rounded-2xl shadow-md p-6 mb-6 transition-all hover:shadow-xl focus-within:ring-2 focus-within:ring-blue-400 outline-none"
      tabIndex={0}
      aria-label={question}
    >
      <div className="text-lg font-semibold text-gray-900 mb-2">{question}</div>
      <div className="text-sm text-gray-500">{hint}</div>
    </motion.div>
  );
}

export default function BehavioralQuestionsPage() {
  const [active, setActive] = useState<number | null>(null);
  const [dailyGoal, setDailyGoal] = useState(1);
  const [checked, setChecked] = useState<{ [key: number]: boolean }>({});
  const today = new Date().toISOString().slice(0, 10);

  useEffect(() => {
    const goals = localStorage.getItem("goals");
    if (goals) setDailyGoal(JSON.parse(goals).dailyBehavioral || 1);
    const progress = localStorage.getItem(`behavioral-progress-${today}`);
    if (progress) setChecked(JSON.parse(progress));
  }, []);

  useEffect(() => {
    localStorage.setItem(
      `behavioral-progress-${today}`,
      JSON.stringify(checked)
    );
  }, [checked, today]);

  return (
    <main className="min-h-screen flex flex-col items-center justify-center bg-gradient-to-br from-blue-50 to-purple-100 py-12 px-4">
      <div className="w-full max-w-2xl mx-auto">
        <h1 className="text-3xl md:text-4xl font-bold text-center mb-8 text-gray-900 drop-shadow-sm">
          Behavioral Questions
        </h1>
        <p className="text-center text-gray-600 mb-10 max-w-xl mx-auto">
          Practice answering common behavioral interview questions. Click a card
          to focus, use hints to structure your answers.
        </p>
        <section aria-label="Behavioral Questions List">
          {questions.slice(0, dailyGoal).map((q, i) => (
            <div
              key={q.id}
              className="relative"
              onClick={() => setActive(q.id)}
            >
              <button
                aria-label={checked[i] ? "Mark as not done" : "Mark as done"}
                onClick={(e) => {
                  e.stopPropagation();
                  setChecked((prev) => ({ ...prev, [i]: !prev[i] }));
                }}
                className={`absolute top-4 right-4 w-8 h-8 rounded-full border-2 flex items-center justify-center transition-all duration-200 shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-400
                  ${
                    checked[i]
                      ? "bg-green-500 border-green-500"
                      : "bg-white border-blue-400 hover:border-blue-600"
                  }
                `}
                style={{ boxShadow: "0 0 0 2px #3b82f6 inset" }}
              >
                {checked[i] ? (
                  <svg
                    className="w-5 h-5 text-white"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      d="M5 13l4 4L19 7"
                    />
                  </svg>
                ) : (
                  <span className="block w-4 h-4 rounded-full border-2 border-blue-400 bg-white" />
                )}
              </button>
              <QuestionCard question={q.question} hint={q.hint} />
            </div>
          ))}
        </section>
      </div>
    </main>
  );
}
