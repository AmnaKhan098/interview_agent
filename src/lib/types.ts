export type User = {
  id: number;
  email: string;
  password: string;
  name: string;
  title?: string;
  location?: string;
  bio?: string;
  avatar?: string;
  created_at?: string;
  updated_at?: string;
};

export type Interview = {
  id: string;
  jobTitle: string;
  date: string;
  status: 'Completed' | 'In Progress';
};

export type Feedback = {
  interviewId: string;
  overallScore: number;
  summary: string;
  questionBreakdown: Array<{
    question: string;
    userAnswer: string;
    feedback: string;
  }>;
};

export type AuthState = {
  user: Omit<User, 'password'> | null;
  isAuthenticated: boolean;
  isLoading: boolean;
};

export type LoginCredentials = {
  email: string;
  password: string;
};

export type RegisterData = {
  name: string;
  email: string;
  password: string;
};