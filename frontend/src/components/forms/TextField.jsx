import * as React from "react";
import "../../App.css";
import TextField from "@mui/material/TextField";
import { Controller } from "react-hook-form";
import { createTheme } from "@mui/material";

const theme = createTheme({
  palette: {
    primary: {
      main: "#facc15",
    },
  },
  typography: {
    fontFamily: ["Poppins", "sans-serif"].join(","),
  },
});

export default function CustomTextField(props) {
  const { label, name, control } = props;
  return (
    <Controller
      name={name}
      control={control}
      render={({
        field: { onChange, value },
        fieldState: { error },
        formState,
      }) => (
        <TextField
          id="outlined-basic"
          onChange={onChange}
          value={value}
          label={label}
          variant="outlined"
          className={"myForm"}
          error={!!error}
          helperText={error?.message}
          sx={{
            padding: "0.5rem",
            marginTop: "0.25rem",
            borderRadius: "0.5rem",
            width: "100%",
            "& .MuiOutlinedInput-root": {
              color: "#FFFFFFFF",
              "& .MuiOutlinedInput-notchedOutline": {
                borderColor: "#F0FF3D",
              },
            },
            "& .MuiInputLabel-outlined": {
              color: "#FFFFFFFF",
              fonteweigth: "bold",
            }
          }}
        />
      )}
    />
  );
}
