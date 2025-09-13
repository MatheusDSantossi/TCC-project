export const routes = {
  business: {
    businessSlug: "/:businessSlug",
  },
  form: {
    login: "/login",
    register: "/register",
    requestPasswordReset: "/request/password-reset",
    passwordResetToken: "/password-reset/:token",
    invitationRegister: "/register/invite/:token",
  },
  admin: {
    adminMain: "/admin",
    adminHome: "/admin-home",
  },
  chatbot: {
    chatbotMain: "/chatbot/:businessSlug",
  },
};
