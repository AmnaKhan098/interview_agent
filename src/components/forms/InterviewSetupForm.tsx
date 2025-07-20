"use client";

import { useState, FormEvent } from "react";
import { useRouter } from "next/navigation";
import { Button } from "../ui/Button";

export function InterviewSetupForm() {
  const router = useRouter();
  const [formData, setFormData] = useState({
    jobTitle: "",
    jobDescription: "",
    techExpertise: "",
    yearsOfExperience: "",
  });

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    // In a real app, we'd create an interview session here
    try {
      const response = await fetch('/api/generate-questions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ jobTitle: formData.jobTitle, jobDescription: formData.jobDescription }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      const generatedQuestions = data.questions;

      // In a real app, we'd create an interview session here
      const createInterviewResponse = await fetch('/api/interview/create', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      });

      if (!createInterviewResponse.ok) {
        throw new Error(`HTTP error! status: ${createInterviewResponse.status}`);
      }

      const interviewData = await createInterviewResponse.json();
      const interviewId = interviewData.interviewId;

      router.push(`/interview/${interviewId}?questions=${encodeURIComponent(JSON.stringify(generatedQuestions))}`);
    } catch (error) {
      console.error('Error generating questions or starting interview:', error);
      // Handle error, e.g., show a message to the user
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-8">
      <div className="space-y-6">
        <div className="space-y-4">
          <label
            htmlFor="jobTitle"
            className="block text-sm font-medium text-gray-700"
          >
            Job Title
          </label>
          <input
            type="text"
            id="jobTitle"
            placeholder="e.g. Senior Frontend Developer"
            value={formData.jobTitle}
            onChange={(e) =>
              setFormData({ ...formData, jobTitle: e.target.value })
            }
            className="w-full rounded-lg border border-gray-300 px-4 py-3 text-gray-900 placeholder-gray-400 focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 transition-colors"
            required
          />
        </div>

        <div className="space-y-4">
          <label
            htmlFor="jobDescription"
            className="block text-sm font-medium text-gray-700"
          >
            Job Description
          </label>
          <textarea
            id="jobDescription"
            placeholder="Paste the job description here..."
            value={formData.jobDescription}
            onChange={(e) =>
              setFormData({ ...formData, jobDescription: e.target.value })
            }
            rows={4}
            className="w-full rounded-lg border border-gray-300 px-4 py-3 text-gray-900 placeholder-gray-400 focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 transition-colors"
            required
          />
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="space-y-4">
            <label
              htmlFor="techExpertise"
              className="block text-sm font-medium text-gray-700"
            >
              Tech Expertise
            </label>
            <input
              type="text"
              id="techExpertise"
              placeholder="e.g. React, Node.js, TypeScript"
              value={formData.techExpertise}
              onChange={(e) =>
                setFormData({ ...formData, techExpertise: e.target.value })
              }
              className="w-full rounded-lg border border-gray-300 px-4 py-3 text-gray-900 placeholder-gray-400 focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 transition-colors"
              required
            />
          </div>

          <div className="space-y-4">
            <label
              htmlFor="yearsOfExperience"
              className="block text-sm font-medium text-gray-700"
            >
              Years of Experience
            </label>
            <input
              type="number"
              id="yearsOfExperience"
              placeholder="0"
              value={formData.yearsOfExperience}
              onChange={(e) =>
                setFormData({ ...formData, yearsOfExperience: e.target.value })
              }
              min="0"
              max="50"
              className="w-full rounded-lg border border-gray-300 px-4 py-3 text-gray-900 placeholder-gray-400 focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 transition-colors"
              required
            />
          </div>
        </div>
      </div>

      <div className="pt-4">
        <Button
          type="submit"
          size="lg"
          className="w-full bg-blue-600 hover:bg-blue-700 text-white font-medium py-3 rounded-lg shadow-sm transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
        >
          Start Interview
        </Button>
      </div>
    </form>
  );
}
