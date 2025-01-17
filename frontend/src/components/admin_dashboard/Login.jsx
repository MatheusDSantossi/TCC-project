import { React, useState } from "react";
import { Box } from "@mui/material";
import CustomTextField from "../forms/TextField";
import CustomPasswordField from "../forms/PasswordField";
import CustomButton from "../forms/Button";
import { Link } from "react-router-dom";
import { useForm } from "react-hook-form";
import AxiosInstance from "../AxiosInstance";
import { useNavigate } from "react-router-dom";
import CustomMessage from "./CustomMessage";

import logo from "../../assets/logo.svg";

import '../../App.css'

const Login = () => {
  const navigate = useNavigate();
  const { handleSubmit, control } = useForm();
  const [ShowMessage, setShowMessage] = useState(false);

  const submission = (data) => {
    AxiosInstance.post(`login/`, {
      email: data.email,
      password: data.password,
    })

      .then((response) => {
        console.log("RESPONSE: ", response);
        localStorage.setItem("Token", response.data.token);
        localStorage.setItem("Email", response.data.user.email);
        navigate(`/admin`);
      })
      .catch((error) => {
        setShowMessage(true);
        console.error("Error during login", error);
      });
  };

  return (
    <div className="flex items-center justify-center min-h-screen p-4">
      {ShowMessage ? (
        <CustomMessage
          text={"Login has failed, please try again, or reset your password"}
          color={"#EC5A76"}
        />
      ) : null}
      <form
        className="w-full max-w-lg p-8 rounded-lg shadow-xl bg-primary"
        onSubmit={handleSubmit(submission)}
      >
        <div className="w-full max-w-lg p-8">
          <div className="flex items-center justify-center mb-5">
            <img className="w-20 mr-4" src={logo} alt="" />
            <h2 className="mb-6 text-2xl font-semibold text-center left-40">Login</h2>
          </div>

          <label className="block text-lg font-semibold text-left text-white">
            Email
          </label>
          <Box className={"itemBox"}>
            <CustomTextField 
            label={"Email"} name={"email"} control={control} />
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

          <div className='flex items-center p-3 space-x-32'>
            <CustomButton
              className="px-4 py-2 mt-4 font-bold text-black transition duration-200 bg-yellow-400 rounded-lg hover:bg-yellow-300"
              label={"Login"}
              type={"submit"}
            />
          </div>
    
          <Box className={"itemBox"} sx={{ flexDirection: "column" }}>
            <Link className='mt-4 text-center text-gray-300' to="/register"> No account yet? <span className='font-medium text-yellow-400 hover:underline hover:text-yellow-300'>Please register!</span> </Link>
            <Link className='mt-4 text-center text-gray-300' to="/request/password_reset">
              {" "}
              Password forgotten? <span className='font-medium text-yellow-400 hover:underline hover:text-yellow-300'>Click here</span>{" "}
            </Link>
          </Box>

        </div>
      </form>
    </div>
  );
};

export default Login;
