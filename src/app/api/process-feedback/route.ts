import { NextRequest, NextResponse } from 'next/server';
import { GoogleGenerativeAI } from '@google/generative-ai';
// import { openDb } from '@/lib/db';
// import { getAuthUser } from '@/lib/auth';
import { cookies } from 'next/headers';
import { getAuthUser } from '@/src/lib/auth';
import { openDb } from '@/src/lib/db';

const GEMINI_API_KEY = process.env.GEMINI_API_KEY;

if (!GEMINI_API_KEY) {
  throw new Error('GEMINI_API_KEY is not set in environment variables');
}

const genAI = new GoogleGenerativeAI(GEMINI_API_KEY);

export async function POST(request: NextRequest) {
  try {
    const user = await getAuthUser(cookies());

    if (!user) {
      return NextResponse.json({ message: 'Unauthorized' }, { status: 401 });
    }

    const { interviewId, questions, answers, jobTitle } = await request.json();

    if (!questions || !answers || questions.length !== answers.length) {
      return NextResponse.json({ error: 'Questions and answers are required and must match in length' }, { status: 400 });
    }

    const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash"});

    const feedbackPromises = questions.map(async (question: string, index: number) => {
      const answer = answers[index];
      const prompt = `Given the following interview question and the candidate's answer, provide a well-crafted feedback and a score out of 10. The feedback should be constructive and highlight areas of improvement.\n\nQuestion: ${question}\nAnswer: ${answer}\n\nFormat the output as a JSON object with 'feedback' (string) and 'score' (number) fields.`;

      const result = await model.generateContent(prompt);
      const response = await result.response;
      const text = response.text();

      try {
        // Attempt to extract JSON from markdown code block if present
        const jsonMatch = text.match(/```json\n([\s\S]*?)\n```/);
        const jsonString = jsonMatch ? jsonMatch[1] : text;
        const parsedFeedback = JSON.parse(jsonString);
        return { question, user_answer: answer, feedback: parsedFeedback.feedback, score: parsedFeedback.score };
      } catch (parseError) {
        console.error('Failed to parse Gemini response. Original text:', text, 'Error:', parseError);
        return { question, user_answer: answer, feedback: `Failed to generate feedback: ${text}`, score: 0 };
      }
    });

    const questionFeedbacks = await Promise.all(feedbackPromises);

    // Calculate overall score and summary
    const totalScore = questionFeedbacks.reduce((sum, fb) => sum + fb.score, 0);
    const overallScore = questionFeedbacks.length > 0 ? totalScore / questionFeedbacks.length : 0;

    // Generate overall summary
    const summaryPrompt = `Given the following interview questions, 
    candidate answers, and individual feedback, provide an overall summary of the 
    candidate's performance and suggest areas for improvement. Focus on the overall 
    strengths and weaknesses.\n\nInterview Details:\nJob Title: ${jobTitle}\n\n${questionFeedbacks.map(fb => `Question: ${fb.question}\nAnswer: ${fb.user_answer}\nFeedback: ${fb.feedback}\nScore: ${fb.score}/10`).join('\n\n')}
    \n\nProvide a concise overall summary.
     Please keep the summary and the feedbacks short`;

    const summaryResult = await model.generateContent(summaryPrompt);
    const summaryText = summaryResult.response.text();
    // Attempt to extract JSON from markdown code block if present for overall summary
    const summaryJsonMatch = summaryText.match(/```json\n([\s\S]*?)\n```/);
    const overallSummary = summaryJsonMatch ? summaryJsonMatch[1] : summaryText;

    return NextResponse.json({
      overallFeedback: overallSummary,
      overallScore: overallScore,
      feedbackResults: questionFeedbacks.map(fb => ({
        question: fb.question,
        feedback: fb.feedback,
        score: fb.score,
      })),
    });
  } catch (error) {
    console.error('Error in process-feedback API:', error);
    return NextResponse.json({ error: 'Failed to process feedback' }, { status: 500 });
  }
}