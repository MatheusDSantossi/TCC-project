const CustomButton = ({
  text,
  icon,
  width = "100%",
  height = "130px",
  paddingTop: padding = "4px",
  rounded = true,
  onClick,
}) => {
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
    >
      {text}
      {icon && { icon }}
    </button>
  );
};

export default CustomButton;
