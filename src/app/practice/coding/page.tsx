"use client";
import React, { useState, useEffect } from "react";
import Link from "next/link";
import { CheckIcon } from "@heroicons/react/24/solid";
import { challenge_data } from "./data";



function CodingChallengeCard({ title, link }: { title: string; link: string }) {
  const [done, setDone] = useState(false);
  return (
    <div
      className={`relative group bg-gray-100 border-2 rounded-2xl shadow-lg p-6 flex flex-col gap-2 transition-all duration-200 hover:shadow-2xl hover:-translate-y-1 hover:border-blue-600 ${
        done ? "border-green-500 bg-green-50" : "border-blue-800"
      }`}
    >
      {/* Custom checkbox toggle in top right */}
      <button
        aria-label={done ? "Mark as not done" : "Mark as done"}
        onClick={() => setDone((v) => !v)}
        className={`absolute top-4 right-4 w-8 h-8 rounded-full border-2 flex items-center justify-center transition-all duration-200 shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-400
          ${
            done
              ? "bg-green-500 border-green-500"
              : "bg-white border-blue-400 hover:border-blue-600"
          }
        `}
        style={{ boxShadow: "0 0 0 2px #3b82f6 inset" }}
      >
        {done ? (
          <CheckIcon className="w-5 h-5 text-white" />
        ) : (
          <span className="block w-4 h-4 rounded-full border-2 border-blue-400 bg-white" />
        )}
      </button>
      <Link
        href={link}
        target="_blank"
        rel="noopener noreferrer"
        className="text-xl font-bold text-blue-800 underline underline-offset-4 hover:text-indigo-700 transition-colors focus:outline-none focus:ring-2 focus:ring-blue-400 rounded"
      >
        {title}
      </Link>
      <div className="mt-1">
        <span className="inline-block bg-blue-100 text-blue-700 text-xs font-semibold px-3 py-1 rounded-full shadow-sm">
          LeetCode
        </span>
      </div>
    </div>
  );
}

export default function CodingChallengesPage() {
  // Placeholder data
  const challenges = challenge_data

  const [dailyGoal, setDailyGoal] = useState(1);
  const [checked, setChecked] = useState<{ [key: number]: boolean }>({});
  const today = new Date().toISOString().slice(0, 10);

  useEffect(() => {
    const goals = localStorage.getItem("goals");
    if (goals) setDailyGoal(JSON.parse(goals).dailyCoding || 1);
    const progress = localStorage.getItem(`coding-progress-${today}`);
    if (progress) setChecked(JSON.parse(progress));
  }, []);

  useEffect(() => {
    localStorage.setItem(`coding-progress-${today}`, JSON.stringify(checked));
  }, [checked, today]);

  return (
    <main className="min-h-screen w-full bg-gradient-to-br from-gray-900 via-blue-950 to-indigo-900 px-4 py-10">
      <header className="mb-10 text-center">
        <h1 className="text-4xl md:text-5xl font-extrabold bg-gradient-to-r from-blue-400 via-indigo-400 to-purple-500 bg-clip-text text-transparent drop-shadow-lg mb-3">
          Coding Challenges
        </h1>
        <p className="text-gray-200 text-lg max-w-2xl mx-auto drop-shadow-sm">
          Sharpen your skills with real interview questions. Pick a challenge
          and start coding!
        </p>
      </header>
      <section>
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-8">
          {challenges.slice(0, dailyGoal).map((c, i) => (
            <div
              key={i}
              className="relative group bg-gray-100 border-2 rounded-2xl shadow-lg p-6 flex flex-col gap-2 transition-all duration-200 hover:shadow-2xl hover:-translate-y-1 hover:border-blue-600"
            >
              <button
                aria-label={checked[i] ? "Mark as not done" : "Mark as done"}
                onClick={() =>
                  setChecked((prev) => ({ ...prev, [i]: !prev[i] }))
                }
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
              <a
                href={c.link}
                target="_blank"
                rel="noopener noreferrer"
                className="text-xl font-bold text-blue-800 underline underline-offset-4 hover:text-indigo-700 transition-colors focus:outline-none focus:ring-2 focus:ring-blue-400 rounded"
              >
                {c.title}
              </a>
              <div className="mt-1">
                <span className="inline-block bg-blue-100 text-blue-700 text-xs font-semibold px-3 py-1 rounded-full shadow-sm">
                  LeetCode
                </span>
              </div>
            </div>
          ))}
        </div>
      </section>
    </main>
  );
}
