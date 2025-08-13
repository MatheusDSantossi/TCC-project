import { ExceptionTypes } from "./ExceptionVariantsTypes";

const ExceptionMainComponent = ({ text, errorMessage, variant }) => {
  return (
    <>
      {variant === ExceptionTypes.ERROR && (
        <div className="absolute text-2xl top-1/2 bg-secondary text-primary">{text}</div>
      )}
      
      {variant === ExceptionTypes.INFO && (
        <div className="text-2xl bg-secondary text-primary">{text}</div>
      )}
      
      {variant === ExceptionTypes.ATTENTION && (
        <div className="text-2xl bg-secondary text-primary">{text}</div>
      )}
    </>
  );
};

export default ExceptionMainComponent;
