"use client";
import React from "react";
import { useRouter } from "next/navigation";

const mockFeedbacks = [
  {
    id: 1,
    date: "2024-05-01",
    interviewType: "Frontend Developer",
    strengths: ["React", "TypeScript", "UI Design"],
    improvements: ["Testing", "State Management"],
  },
  {
    id: 2,
    date: "2024-04-20",
    interviewType: "Backend Developer",
    strengths: ["Node.js", "API Design"],
    improvements: ["Database Optimization"],
  },
  {
    id: 3,
    date: "2024-03-15",
    interviewType: "Fullstack Developer",
    strengths: ["Problem Solving", "Communication"],
    improvements: ["System Design"],
  },
];

export default function AllFeedbacksPage() {
  const router = useRouter();
  return (
    <div className="min-h-screen bg-gradient-to-b from-blue-50 to-purple-50 py-12 px-4 flex flex-col items-center">
      <div className="max-w-3xl w-full">
        <h1 className="text-4xl font-extrabold text-blue-900 mb-8 text-center drop-shadow-sm">
          All Interview Feedbacks
        </h1>
        <div className="space-y-6">
          {mockFeedbacks.map((fb) => (
            <div
              key={fb.id}
              className="bg-white/90 border border-blue-100 rounded-2xl shadow-lg p-6 flex flex-col md:flex-row md:items-center md:justify-between transition hover:shadow-xl"
            >
              <div>
                <div className="flex items-center gap-3 mb-2">
                  <span className="text-lg font-semibold text-blue-700">
                    {fb.interviewType}
                  </span>
                  <span className="text-xs bg-blue-100 text-blue-700 rounded px-2 py-0.5">
                    {fb.date}
                  </span>
                </div>
                <div className="flex flex-wrap gap-4 text-sm">
                  <div>
                    <span className="font-medium text-green-700">
                      Strengths:
                    </span>
                    <ul className="list-disc list-inside text-gray-800 ml-4">
                      {fb.strengths.map((s, i) => (
                        <li key={i}>{s}</li>
                      ))}
                    </ul>
                  </div>
                  <div>
                    <span className="font-medium text-red-700">
                      Improvements:
                    </span>
                    <ul className="list-disc list-inside text-gray-800 ml-4">
                      {fb.improvements.map((imp, i) => (
                        <li key={i}>{imp}</li>
                      ))}
                    </ul>
                  </div>
                </div>
              </div>
              <button
                className="mt-4 md:mt-0 bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-5 rounded-lg shadow transition-colors text-sm"
                onClick={() => router.push(`/feedback`)}
              >
                View Details
              </button>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
