const CustomButton = ({
  text,
  icon,
  width = "100%",
  height = "130px",
  paddingTop = "4px",
  rounded = true,
}) => {
  return (
    <button
      className={`bg-primary hover:bg-button-hover-green p-[${paddingTop}] w-[${width}] h-[${height}] ${
        rounded ? "rounded" : ""
      }`}
    >
      {text}
      {icon && { icon }}
    </button>
  );
};

export default CustomButton;
