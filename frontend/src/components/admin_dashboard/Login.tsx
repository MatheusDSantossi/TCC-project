import { useState } from "react";
import { Box } from "@mui/material";
import CustomTextField from "../forms/TextField";
import CustomPasswordField from "../forms/PasswordField";
import { Link } from "react-router-dom";
import { useForm } from "react-hook-form";
import AxiosInstance from "../AxiosInstance";
import { useNavigate } from "react-router-dom";
import CustomMessage from "./CustomMessage";

import logo from "../../assets/logo.svg";
import googleBrand from "../../assets/brand_google.svg";

import "../../App.css";

import ReactiveButton from "reactive-button";
import GoogleButton from "./GoogleButton";
import { routes } from "../../routes/routes";

const Login = () => {
  // routes
  const requestPasswordReset = routes.form.requestPasswordReset;
  const registerRoute = routes.form.register;
  const adminMainRoute = routes.admin.adminMain;

  const navigate = useNavigate();
  const { handleSubmit, control } = useForm();
  const [showMessage, setShowMessage] = useState(false);
  const [buttonState, setButtonState] = useState("idle");

  const submission = (data) => {
    setButtonState("loading");

    AxiosInstance.post(`login/`, {
      email: data.email,
      password: data.password,
    })

      .then((response) => {
        setButtonState("success");
        const token = response.data.token;
        const biz = response.data.user?.business || response.data.business;

        localStorage.setItem("Email", response.data.email);
        localStorage.setItem("Token", token);
        localStorage.setItem("BusinessSlug", biz);
        // navigate(`/admin`);
        navigate(adminMainRoute);
      })
      .catch((error) => {
        setButtonState("error");
        setShowMessage(true);
        console.error("Error during login", error);
      });
  };

  const googleLoginHandle = () => {
    window.location.href = "http://localhost:8000/accounts/google/login";
  };

  const handleClose = (_, reason) => {
    if (reason === "clickaway") return;
    setShowMessage(false);
  };

  return (
    <div className="flex items-center justify-center min-h-screen p-4 drop-shadow-3xl">
      {showMessage ? (
        <CustomMessage
          open={showMessage}
          onClose={handleClose}
          severity="error"
          message={
            "Login falhou, por favor tente novamente, ou resete sua senha."
          }
          duration={3000}
        />
      ) : null}
      <form
        className="w-full max-w-lg p-8 rounded-lg shadow-xl bg-primary"
        onSubmit={handleSubmit(submission)}
      >
        <div className="w-full max-w-lg p-8">
          <div className="flex items-center justify-center mb-5">
            <img className="w-20 mr-4" src={logo} alt="" />
            <h2 className="mb-6 text-2xl font-semibold text-center left-40">
              Login
            </h2>
          </div>
          <Box className={"itemBox mb-3"}>
            <CustomTextField
              label={"Email"}
              name={"email"}
              control={control}
              size={"small"}
            />
          </Box>

          <Box className={"itemBox"}>
            <CustomPasswordField
              label={"Senha"}
              name={"password"}
              size={"small"}
              control={control}
            />
          </Box>

          <div className="flex items-center justify-center p-3 space-x-16">
            {/* <CustomButton
              className="px-4 py-2 mt-4 font-bold text-black transition duration-200 bg-yellow-400 rounded-lg hover:bg-yellow-300"
              label={"Login"}
              type={"submit"}
            /> */}
            <ReactiveButton
              className="text-primary hover:bg-yellow-300"
              buttonState={buttonState}
              color="yellow"
              type={"submit"}
              loadingText="Carregando..."
              successText={"Logado com sucesso"}
              idleText={"Login"}
              animation="true"
              rounded="true"
              width={200}
              height={40}
              size="large"
            />
            {/* </div>

          <div className="flex items-center justify-center m-3"> */}
            {/* TODO: I need to fix the problem witht the login using google */}
            {/* <GoogleButton /> */}

            {/* <button className="flex items-center p-1 rounded rounded-full bg-[#FFC107] bg-white text-black hover:bg-yellow-300" onClick={googleLoginHandle}>
              <img className="h-10" src={googleBrand} alt="Logo do Google" />
              Login com Google
            </button> */}
            {/* <a className="p-2 text-black bg-[#FFC107] rounded-full hover:bg-yellow-300" href="http://localhost:8000/accounts/google/login">
              Login com Google
            </a> */}
          </div>

          <Box className={"itemBox"} sx={{ flexDirection: "column" }}>
            <Link className="mt-4 text-center text-gray-300" to={registerRoute}>
              {" "}
              Não possui conta?{" "}
              <span className="font-medium text-yellow-400 hover:underline hover:text-yellow-300">
                Por favor se cadastre!
              </span>{" "}
            </Link>
            <br />
            <Link
              className="mt-4 text-center text-gray-300"
              to={requestPasswordReset}
            >
              {" "}
              Esqueceu a senha?{" "}
              <span className="font-medium text-yellow-400 hover:underline hover:text-yellow-300">
                Clique aqui
              </span>{" "}
            </Link>
          </Box>
        </div>
      </form>
    </div>
  );
};

export default Login;
