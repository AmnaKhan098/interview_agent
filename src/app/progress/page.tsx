"use client";
import { FaCheckCircle, FaHourglassHalf, FaTrophy } from "react-icons/fa";
import { useEffect, useState } from "react";

export default function ProgressPage() {
  const [goals, setGoals] = useState({
    interviewDate: "",
    dailyInterviews: 1,
    dailyCoding: 1,
    dailyBehavioral: 1,
  });

  useEffect(() => {
    const stored = localStorage.getItem("goals");
    if (stored) setGoals(JSON.parse(stored));
  }, []);

  // Dummy data, replace with real data as needed
  const progress = 65;
  const completed = 12;
  const inProgress = 3;
  // const goals = 20; // replaced

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex flex-col items-center py-12 px-4">
      <div className="w-full max-w-2xl bg-white rounded-3xl shadow-2xl p-8">
        <h1 className="text-4xl font-extrabold text-indigo-700 mb-2 text-center">
          Your Progress
        </h1>
        <p className="text-gray-500 text-center mb-8">
          Track your interview prep journey
        </p>
        {goals.interviewDate && (
          <div className="mb-4 text-center text-lg text-blue-700 font-semibold">
            Interview Date: {goals.interviewDate}
          </div>
        )}
        <div className="mb-8">
          <div className="flex justify-between mb-2">
            <span className="text-lg font-semibold text-indigo-600">
              Overall Progress
            </span>
            <span className="text-lg font-semibold text-indigo-600">
              {progress}%
            </span>
          </div>
          <div className="w-full bg-indigo-100 rounded-full h-4">
            <div
              className="bg-gradient-to-r from-indigo-500 to-blue-400 h-4 rounded-full transition-all duration-500"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-6">
          <div className="flex flex-col items-center bg-[#f3f6fd] rounded-xl p-6 shadow">
            <FaCheckCircle className="text-4xl text-green-500 mb-2" />
            <span className="text-2xl font-extrabold text-gray-700 mb-1">
              {completed} / {goals.dailyCoding}
            </span>
            <span className="text-gray-500 font-medium">Coding (Today)</span>
          </div>
          <div className="flex flex-col items-center bg-[#f3f6fd] rounded-xl p-6 shadow">
            <FaHourglassHalf className="text-4xl text-yellow-500 mb-2" />
            <span className="text-2xl font-extrabold text-gray-700 mb-1">
              {inProgress} / {goals.dailyBehavioral}
            </span>
            <span className="text-gray-500 font-medium">
              Behavioral (Today)
            </span>
          </div>
          <div className="flex flex-col items-center bg-[#f3f6fd] rounded-xl p-6 shadow">
            <FaTrophy className="text-4xl text-blue-600 mb-2" />
            <span className="mb-1">
              <span className="inline-block bg-blue-500 text-white text-2xl font-extrabold rounded-lg px-3 py-1">
                {goals.dailyInterviews}
              </span>
            </span>
            <span className="text-gray-500 font-medium">
              Interviews (Daily Goal)
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
