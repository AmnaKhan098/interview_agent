import { NextRequest, NextResponse } from 'next/server';
import { GoogleGenerativeAI } from '@google/generative-ai';

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY as string);

export async function POST(request: NextRequest) {
  try {
    const { questions, answers } = await request.json();

    if (!questions || !answers || questions.length !== answers.length) {
      return NextResponse.json({ error: 'Questions and answers are required and must match in length' }, { status: 400 });
    }

    const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash"});

    const feedbackPromises = questions.map(async (question: string, index: number) => {
      const answer = answers[index];
      const prompt = `Given the following interview question and the candidate's answer, provide a well-crafted feedback and a score out of 10. The feedback should be constructive and highlight areas of improvement.

Question: ${question}
Answer: ${answer}

Format the output as a JSON object with 'feedback' (string) and 'score' (number) fields. 
keep the feedback short`;

      const result = await model.generateContent(prompt);
      const response = await result.response;
      const text = response.text();

      try {
        return JSON.parse(text);
      } catch (parseError) {
        console.error('Failed to parse Gemini response for feedback:', text, parseError);
        return { feedback: 'Could not generate feedback for this answer.', score: 0 };
      }
    });

    const feedbackResults = await Promise.all(feedbackPromises);

    return NextResponse.json({ feedbackResults });
  } catch (error) {
    console.error('Error generating feedback:', error);
    return NextResponse.json({ error: 'Failed to generate feedback' }, { status: 500 });
  }
}