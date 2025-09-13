import { ExceptionTypes } from "./ExceptionVariantsTypes";
import { MdContentCopy } from "react-icons/md";
import CustomButton from "../components/custom/CustomButton";
import { IoMdCloseCircleOutline } from "react-icons/io";
import { FaClipboardCheck } from "react-icons/fa6";
import { useState } from "react";

const ExceptionMainComponent = ({ text, error, variant }) => {
  // console.log("error: ", error)
  // State
  const extendErrorMessage = error.message + " " + error.code;
  const [copied, setCopied] = useState(false);
  const [display, setDisplay] = useState("flex");

  const handleCopyClick = async () => {
    try {
      await navigator.clipboard.writeText(extendErrorMessage);
      setCopied(true);
      // Reset copied state after a short delay for visual feedback
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error("Failed to copy text: ", err);
    }
  };

  return (
    <div
      key={display}
      className="mx-auto absolute text-[14px]"
      style={{ display: display }}
    >
      {variant === ExceptionTypes.ERROR && (
        <div className="flex flex-col justify-start w-[500px] h-[150px] px-6 pb-46 pt-4 mx-auto rounded border-2 border-error bg-gray-100 shadow-2xl">
          <header className="text-base text-start">
            <IoMdCloseCircleOutline className="text-error" size={40} />
            <h1 className="text-primary pb-3">Error: {text}</h1>
            {/* <h3 className="text-primary">{text}</h3> */}
          </header>
          <div className="flex flex-col p-3 bg-white border-2 top-1/2 border-primary mb-4">
            {copied ? (
              <FaClipboardCheck
                className="items-end justify-end relative cursor-pointer text-primary hover:text-secondary-dark"
                size={16}
              />
            ) : (
              <MdContentCopy
                className="items-end justify-end relative cursor-pointer text-primary hover:text-secondary-dark"
                onClick={handleCopyClick}
                size={16}
              />
            )}
            <p className="absolute ml-10 text-[#9E9E9E]">
              {extendErrorMessage}{" "}
            </p>
          </div>
          <CustomButton
            width="100px"
            text={"Ok"}
            type="submit"
            onClick={() => {
              setDisplay("none");
            }}
          />
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
