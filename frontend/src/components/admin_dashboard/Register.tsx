import { Box } from "@mui/material";
import CustomTextField from "../forms/TextField";
import CustomPasswordField from "../forms/PasswordField";
import { Link } from "react-router-dom";
import { useForm } from "react-hook-form";
import AxiosInstance from "../AxiosInstance";
import { useNavigate } from "react-router-dom";
import { yupResolver } from "@hookform/resolvers/yup";
import * as yup from "yup";

import logo from "../../assets/logo.svg";

import ReactiveButton from "reactive-button";
import { useState } from "react";
import { routes } from "../../routes/routes";

const Register = () => {
  // routes
  const loginRoute = routes.form.login;
  const adminMainRoute = routes.admin.adminMain;

  const navigate = useNavigate();
  const [buttonState, setButtonState] = useState("idle");

  const schema = yup.object({
    email: yup
      .string()
      .email("Esse campo espera um e-mail válido")
      .required("Email é um campo obrigatório"),
    password: yup
      .string()
      .required("Senha é um campo obrigatório")
      .min(8, "Senha deve ter pelo menos 8 caracteres")
      .matches(/[A-Z]/, "Senha deve ter pelo menos uma letra maiúscula")
      .matches(/[a-z]/, "Senha deve ter pelo menos uma letra minuscúla")
      .matches(/[0-9]/, "Senha deve ter pelo menos um número")
      .matches(
        /[!@#$%^&*(),.?":;{}|<>+]/,
        "Senha deve ter pelo menos um caracter especial"
      ),
    password2: yup
      .string()
      .required("O campo de confirmação de senha é obrigatorio")
      .oneOf([yup.ref("password"), null], "As senhas devem ser iguais"),
  });

  const { handleSubmit, control } = useForm({ resolver: yupResolver(schema) });

  const submission = (data) => {
    setButtonState("success");

    AxiosInstance.post(`register/`, {
      email: data.email,
      password: data.password,
    }).then(() => {
      setTimeout(() => {
        setButtonState("error");
      }, 2000);
      navigate(adminMainRoute);
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
              Cadastro
            </h2>
          </div>

          <Box className={"itemBox"}>
            <CustomTextField
              label={"Email"}
              name={"email"}
              control={control}
              size={"small"}
            />
          </Box>

          <Box className={"itemBox "}>
            <CustomPasswordField
              label={"Senha"}
              name={"password"}
              control={control}
              size={"small"}
            />
          </Box>

          <Box className={"itemBox"}>
            <CustomPasswordField
              label={"Confirmar Senha"}
              name={"password2"}
              control={control}
              size={"small"}
            />
          </Box>

          {/* <a href="http://127.0.0.1:8000/accounts/google/login">
            Cadastrar com Google
          </a> */}

          <div className="flex items-center justify-center p-3 space-x-32">
            {/* <CustomButton
                            type={"submit"}
                            label={"Register"}
                        /> */}
            <ReactiveButton
              className="font-bold text-primary hover:bg-yellow-300"
              buttonState={buttonState}
              color="yellow"
              type={"submit"}
              loadingText="Loading..."
              successText={"Cadastrado com sucesso"}
              idleText={"Cadastrar"}
              animation={true}
              rounded={true}
              width={80}
              height={"40"}
              size="large"
            />
          </div>

          <Box className={"itemBox"}>
            <Link className="mt-4 text-center text-gray-300" to={loginRoute}>
              {" "}
              Já possui cadastro?{" "}
              <span className="font-medium text-yellow-400 hover:underline hover:text-yellow-300">
                Por favor, faça login!
              </span>{" "}
            </Link>
          </Box>
        </Box>
      </form>
    </div>
  );
};

export default Register;
