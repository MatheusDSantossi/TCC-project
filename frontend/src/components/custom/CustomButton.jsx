const CustomButton = ({
  text,
  icon,
  width = "100%",
  height = "130px",
  paddingTop: padding = "4px",
  rounded = true,
}) => {
  return (
    <button
      className={`bg-primary hover:bg-transparent hover:border hover:border-primary hover:text-primary ${
        rounded ? "rounded" : ""
      }`}
      style={{
        width: width,
        height: height,
        padding: padding
      }}
    >
      {text}
      {icon && { icon }}
    </button>
  );
};

export default CustomButton;
