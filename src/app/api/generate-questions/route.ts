import { GoogleGenerativeAI } from '@google/generative-ai';
import { NextResponse } from 'next/server';

const GEMINI_API_KEY = process.env.GEMINI_API_KEY;

if (!GEMINI_API_KEY) {
  throw new Error('GEMINI_API_KEY is not set in environment variables');
}

const genAI = new GoogleGenerativeAI(GEMINI_API_KEY);

export async function POST(request: Request) {
  try {
    const { jobTitle, jobDescription } = await request.json();

    if (!jobTitle || !jobDescription) {
      return NextResponse.json({ error: 'Job title and description are required' }, { status: 400 });
    }

    const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash"});

    const prompt = `Generate 5 interview questions for a candidate based on the following job title and description:\n\nJob Title: ${jobTitle}\nJob Description: ${jobDescription}\n\nProvide a list of questions suitable for a technical interview. Retrun only the interview questions. Do not add any extra text`;

    const result = await model.generateContent(prompt);
    const response = await result.response;
    const text = response.text();

    // Assuming the model returns questions as a list, parse them.
    // This parsing might need adjustment based on the actual model output format.
    const questions = text.split('\n').filter(q => q.trim() !== '');

    return NextResponse.json({ questions });
  } catch (error) {
    console.error('Error generating questions:', error);
    return NextResponse.json({ error: 'Failed to generate questions' }, { status: 500 });
  }
}