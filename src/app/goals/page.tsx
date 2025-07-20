"use client";
import { useState } from "react";
import { Button } from "../../components/ui/Button";

export default function GoalsPage() {
  const [interviewDate, setInterviewDate] = useState("");
  const [dailyInterviews, setDailyInterviews] = useState(1);
  const [dailyCoding, setDailyCoding] = useState(1);
  const [dailyBehavioral, setDailyBehavioral] = useState(1);
  const [saved, setSaved] = useState(false);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const data = {
      interviewDate,
      dailyInterviews,
      dailyCoding,
      dailyBehavioral,
    };
    localStorage.setItem("goals", JSON.stringify(data));
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  return (
    <main className="min-h-screen flex flex-col items-center justify-center bg-gradient-to-br from-blue-50 to-purple-100 py-12 px-4">
      <div className="w-full max-w-lg bg-white rounded-3xl shadow-2xl p-8">
        <h1 className="text-3xl font-bold text-indigo-700 mb-2 text-center">
          Set Your Goals
        </h1>
        <p className="text-gray-500 text-center mb-8">
          Plan your interview prep journey
        </p>
        <form className="space-y-6" onSubmit={handleSubmit}>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Interview Date
            </label>
            <input
              type="date"
              value={interviewDate}
              onChange={(e) => setInterviewDate(e.target.value)}
              required
              className="w-full rounded-lg border border-gray-300 px-4 py-3 text-gray-900 focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 transition-colors"
            />
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Daily Interviews
              </label>
              <input
                type="number"
                min={1}
                value={dailyInterviews}
                onChange={(e) => setDailyInterviews(Number(e.target.value))}
                className="w-full rounded-lg border border-gray-300 px-4 py-2"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Daily Coding
              </label>
              <input
                type="number"
                min={1}
                value={dailyCoding}
                onChange={(e) => setDailyCoding(Number(e.target.value))}
                className="w-full rounded-lg border border-gray-300 px-4 py-2"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Daily Behavioral
              </label>
              <input
                type="number"
                min={1}
                value={dailyBehavioral}
                onChange={(e) => setDailyBehavioral(Number(e.target.value))}
                className="w-full rounded-lg border border-gray-300 px-4 py-2"
              />
            </div>
          </div>
          <Button
            type="submit"
            className="w-full bg-blue-600 hover:bg-blue-700 text-white font-medium py-3 rounded-lg shadow-sm"
          >
            Save Goals
          </Button>
          {saved && (
            <div className="text-green-600 text-center mt-2">Goals saved!</div>
          )}
        </form>
      </div>
    </main>
  );
}
