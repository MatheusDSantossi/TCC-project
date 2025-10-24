import type { JSXElementConstructor, ReactElement, ReactNode } from "react";

interface customButtonProps {
  text?: string;
  icon?: ReactNode;
  width?: string;
  height?: string;
  paddingTop?: string;
  tooltipContent?: string | ReactNode;
  tooltipPosition?: "right" | "left" | "top" | "bottom";
  rounded?: boolean;
  onClick?: () => void;
  type?: string;
}

const CustomButton = ({
  text,
  icon,
  width = "100%",
  height = "130px",
  paddingTop: padding = "4px",
  rounded = true,
  tooltipContent = "Copiar",
  tooltipPosition = "top",
  onClick,
  type,
}: customButtonProps) => {
  return (
    <button
      className={`bg-primary hover:bg-transparent hover:border hover:border-primary hover:text-primary ${
        rounded ? "rounded" : ""
      }`}
      style={{
        width: width,
        height: height,
        padding: padding,
      }}
      onClick={onClick}
      data-tooltip-id="custom-button-tooltip"
      data-tooltip-content={tooltipContent}
      data-tooltip-place={tooltipPosition}
    >
      {text}
      {icon && <div>{icon}</div>}
    </button>
  );
};

export default CustomButton;
