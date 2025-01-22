import { Box } from "@mui/material";
import CustomTextField from "../forms/TextField";
import CustomPasswordField from "../forms/PasswordField";
import CustomButton from "../forms/Button";
import { Link } from "react-router-dom";
import { useForm } from "react-hook-form";
import AxiosInstance from "../AxiosInstance";
import { useNavigate } from "react-router-dom";
import { yupResolver } from "@hookform/resolvers/yup";
import * as yup from "yup";

import logo from "../../assets/logo.svg";

import ReactiveButton from "reactive-button";
import { useState } from "react";

const Register = () => {
  const navigate = useNavigate();
  const [buttonState, setButtonState] = useState("idle");

  const schema = yup.object({
    email: yup
      .string()
      .email("Field expects an email adress")
      .required("Email is a required field"),
    password: yup
      .string()
      .required("Password is a required field")
      .min(8, "Password must be at least 8 characters")
      .matches(/[A-Z]/, "Password must contain at least one uppercase letter")
      .matches(/[a-z]/, "Password must contain at least one lower case letter")
      .matches(/[0-9]/, "Password must contain at least one number")
      .matches(
        /[!@#$%^&*(),.?":;{}|<>+]/,
        "Password must contain at least one special character"
      ),
    password2: yup
      .string()
      .required("Password confirmation is a required field")
      .oneOf([yup.ref("password"), null], "Passwords must match"),
  });

  const { handleSubmit, control } = useForm({ resolver: yupResolver(schema) });

  const submission = (data) => {
    setButtonState("loading");

    AxiosInstance.post(`register/`, {
      email: data.email,
      password: data.password,
    }).then(() => {
      setTimeout(() => {
        setButtonState("success");
      }, 2000);
      navigate(`/admin`);
    });
  };

  return (
    <div className="flex items-center justify-center min-h-screen p-4">
      <form
        onSubmit={handleSubmit(submission)}
        className="w-full max-w-lg p-8 rounded-lg shadow-xl bg-primary"
      >
        <Box className={"whiteBox"}>
          <div className="flex items-center justify-center mb-5">
            <img className="w-20 mr-4" src={logo} alt="" />
            <h2 className="mb-6 text-2xl font-semibold text-center left-40">
              {" "}
              User registration{" "}
            </h2>
          </div>

          <label className="block text-lg font-semibold text-left text-white">
            Email
          </label>
          <Box className={"itemBox"}>
            <CustomTextField label={"Email"} name={"email"} control={control} />
          </Box>

          <label className="block text-lg font-semibold text-left text-white">
            Password
          </label>
          <Box className={"itemBox"}>
            <CustomPasswordField
              label={"Password"}
              name={"password"}
              control={control}
            />
          </Box>

          <label className="block text-lg font-semibold text-left text-white">
            Confirm password
          </label>
          <Box className={"itemBox"}>
            <CustomPasswordField
              label={"Confirm password"}
              name={"password2"}
              control={control}
            />
          </Box>
          <div className="flex items-center justify-center p-3 space-x-32">
            {/* <CustomButton
                            type={"submit"}
                            label={"Register"}
                        /> */}
            <ReactiveButton
              className="text-primary hover:bg-yellow-300"
              buttonState={buttonState}
              color="yellow"
              type={"submit"}
              loadingText="Loading..."
              successText={"Registered successful"}
              idleText={"Register"}
              animation="true"
              rounded="true"
              width={80}
              height={40}
              size="large"
            />
          </div>

          <Box className={"itemBox"}>
            <Link className="mt-4 text-center text-gray-300" to="/login">
              {" "}
              Already registered?{" "}
              <span className="font-medium text-yellow-400 hover:underline hover:text-yellow-300">
                Please login!
              </span>{" "}
            </Link>
          </Box>
        </Box>
      </form>
    </div>
  );
};

export default Register;
