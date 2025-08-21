import { Button } from "@material-tailwind/react";
import { ExceptionTypes } from "./ExceptionVariantsTypes";
import { MdContentCopy } from "react-icons/md";
import CustomButton from "../components/custom/CustomButton";

const ExceptionMainComponent = ({ text, error, variant }) => {
  return (
    <div className="mx-auto relative">
      {variant === ExceptionTypes.ERROR && (
        <div className="flex flex-col justify-start w-[500px] h-[150px] px-6 py-6 mx-auto rounded bg-error-background shadow-2xl">
          <header className="text-xl text-start">
            <h1 className="text-white">Error:</h1>
            <h3 className="text-white">{error}</h3>
          </header>
          <div className="flex flex-col p-3 bg-white border-2 top-1/2 border-primary mb-4">
            <MdContentCopy className="items-end justify-end relative" size={16} color="#012921" />
            <p className="absolute ml-10 text-base text-[#9E9E9E]">{text}</p>
          </div>

          <CustomButton text={"Ok"} type="submit" />
        </div>
      )}

      {variant === ExceptionTypes.INFO && (
        <div className="text-2xl bg-secondary text-primary">{text}</div>
      )}

      {variant === ExceptionTypes.ATTENTION && (
        <div className="text-2xl bg-secondary text-primary">{text}</div>
      )}
    </div>
  );
};

export default ExceptionMainComponent;
