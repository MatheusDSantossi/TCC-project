export enum ExceptionVariant {
  ERROR = "error",
  INFO = "info",
  CONFIRMATION = "confirmation",
};

export interface ExceptionPayload {
  text: string; 
  error?: any;
  variant?: string;
}
