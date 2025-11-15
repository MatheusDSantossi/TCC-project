import {
  ExceptionVariant,
  type ExceptionPayload,
} from "./ExceptionVariantsTypes";
import { MdContentCopy } from "react-icons/md";
import CustomButton from "../components/custom/CustomButton";
import { IoMdCloseCircleOutline } from "react-icons/io";
import { FaClipboardCheck } from "react-icons/fa6";
import { IoInformationCircleSharp } from "react-icons/io5";
import { BsFillQuestionCircleFill } from "react-icons/bs";
import { useState } from "react";

const ExceptionMainComponent = ({
  text,
  error,
  variant,
  onClose,
}: ExceptionPayload) => {
  const handleOk = () => {
    onClose?.();
  };

  // console.log("error: ", error)
  // State
  const extendErrorMessage = error.message + " " + error.code;
  const [copied, setCopied] = useState(false);

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
      className="fixed inset-0 z-[1000] flex items-center justify-center text-[14px]"
    >
      {/* Backdrop */}
      {/* <div className="absolute inset-0 bg-black/50" onClick={handleOk} /> */}
      <div className="absolute inset-0 bg-black/50" />

      {/* Modal card */}
      <div
        // responsive width, limited height, scroll inside if content too tall
        className={`relative z-10 w-full max-w-lg mx-4 bg-gray-100 rounded-l-2xl shadow-2xl border-3 max-h-[80vh] overflow-y-auto ${variant === ExceptionVariant.ERROR ? "border-error" : variant === ExceptionVariant.INFO ? "border-info" : "border-confirmation"}`}
        role="dialog"
      >
        {/* Padding and layout inside */}
        <div className="p-6">
          {/* Header */}
          <header className="flex items-center gap-3 mb-4">
            {variant === ExceptionVariant.ERROR && (
              <IoMdCloseCircleOutline className="text-error" size={40} />
            )}
            {variant === ExceptionVariant.INFO && (
              <IoInformationCircleSharp className="text-info" size={40} />
            )}
            {variant === ExceptionVariant.CONFIRMATION && (
              <BsFillQuestionCircleFill
                className="text-confirmation"
                size={40}
              />
            )}
            <h2 className="text-lg font-semibold text-primary">
              {variant === ExceptionVariant.ERROR
                ? "Error: "
                : variant === ExceptionVariant.INFO
                ? "Info: "
                : "Confirmation: "}
              <span className="font-normal ml-1">{text}</span>
            </h2>
          </header>

          {/* Body: small box with copy button and message */}
          <div className="relative flex items-start gap-3 bg-white border rounded p-3 mb-4 shadow">
            <div className="flex shrink-0">
              {copied ? (
                <FaClipboardCheck
                  className="text-primary cursor-pointer"
                  size={18}
                />
              ) : (
                <MdContentCopy
                  className="text-primary cursor-pointer hover:text-secondary-dark"
                  size={18}
                  onClick={handleCopyClick}
                />
              )}
            </div>

            <div className="text-sm text-gray-600 break-words">
              {/* show extended error message; trimmed if too long */}
              {extendErrorMessage}
            </div>
          </div>

          {/* Footer: actions */}
          <div className="flex justify-end gap-3">
            <CustomButton
              width="80px"
              height="35px"
              text="Ok"
              type="button"
              onClick={handleOk}
            />
          </div>
        </div>
      </div>
    </div>
  );
};

export default ExceptionMainComponent;
