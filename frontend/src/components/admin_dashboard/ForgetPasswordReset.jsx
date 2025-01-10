import { React, useState } from 'react'
import { Box } from '@mui/material'
import CustomTextField from '../forms/TextField'
import CustomPasswordField from '../forms/PasswordField'
import CustomButton from '../forms/Button'
import { Link } from 'react-router-dom'
import { useForm } from 'react-hook-form'
import AxiosInstance from '../AxiosInstance'
import { useNavigate, useParams } from 'react-router-dom'
import CustomMessage from './CustomMessage'

const ForgetPasswordReset = () => {
    const navigate = useNavigate()
    const { handleSubmit, control } = useForm()
    const { token } = useParams()
    console.log(token)

    const [ShowMessage, setShowMessage] = useState(false)

    const submission = (data) => {
        AxiosInstance.post(`api/password_reset/confirm/`, {
            password: data.password,
            token: token,
        })

            .then((response) => {
                setShowMessage(true)
                setTimeout(() => {
                   navigate('/login') 
                }, 6000);
            })

    }
    return (
        <div className={"myBackground"}>

            {ShowMessage ? <CustomMessage text={"Your password reset was successfully. You will be directed to the login page in a second"} color={'#69C9AB'} /> : null}
            <form onSubmit={handleSubmit(submission)}>



                <Box className={"whiteBox"}>

                    <Box className={"itemBox"}>
                        <Box className={"title"}> Reset password </Box>
                    </Box>

                    <Box className={"itemBox"}>
                        <CustomPasswordField
                            label={"Password"}
                            name={"password"}
                            control={control}
                        />
                    </Box>

                    <Box className={"itemBox"}>
                        <CustomPasswordField
                            label={"Confirm Password"}
                            name={"password2"}
                            control={control}
                        />
                    </Box>

                    
                    <Box className={"itemBox"}>
                        <CustomButton
                            label={" Reset password"}
                            type={"submit"}
                        />
                    </Box>

                    <Box className={"itemBox"} sx={{ flexDirection: 'column' }}>

                    </Box>


                </Box>

            </form>

        </div>
    )

}

export default ForgetPasswordReset