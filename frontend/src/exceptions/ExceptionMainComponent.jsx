import { Button } from "@material-tailwind/react";
import { ExceptionTypes } from "./ExceptionVariantsTypes";
import { MdContentCopy } from "react-icons/md";

const ExceptionMainComponent = ({ text, error, variant }) => {
  return (
    <>
      {variant === ExceptionTypes.ERROR && (
        <div className="flex flex-col justify-center px-6 pb-16 mx-auto rounded bg-error-background">
          <header className="text-xl text-start">
            <h1 className="text-black">Error:</h1>
            <h3 className="text-black">{error}</h3>
          </header>
          <div className="flex flex-col p-2 bg-white border-2 top-1/2 border-secondary">
            <MdContentCopy className="items-end justify-end" size={16} color="#012921" />
            <p className="text-base text-[#9E9E9E]">{text}</p>
          </div>
          <Button label={"Ok"} type="submit" />
        </div>
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
