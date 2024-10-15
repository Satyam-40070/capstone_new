import User from '../models/userModel.js';
import bcrypt from 'bcryptjs'
import jwt from 'jsonwebtoken';
import dotenv from 'dotenv';
dotenv.config();

export const Login = async (req,res)=>{
    try{
        const {email, password} = req.body;
        if(!(email && password)){
            return res.status(400).json({message: "Please fill all fields"});
        }

        const user = await User.findOne({email});
        if(!user){
            return res.status(401).send("User does not exist");
        }

        const enteredPassword = await bcrypt.compare(password, user.password);
        if(!enteredPassword){
            return res.status(401).send("Incorrect password");
        }
        
        const token = jwt.sign({ id: user._id , role:user.role}, process.env.SECRET_KEY, {
            expiresIn: "1d",
        });
        user.token = token;
        user.password = undefined;

        //store cookies
        const options = {
            expires: new Date(Date.now() + 1 * 24 * 60 * 60 * 1000),
            httpOnly: true, //only manipulate by server not by client/user
        };

        //send the token
        res.status(200).cookie("token", token, options).json({
            message: "You have successfully logged in!",
            success: true,
            token,
        });

    }catch(error){
        console.log(error.message);
    }
};

export const logout = (req, res) => {
    res.clearCookie("token",{path:'/'});
    res.status(200).json({ message: "You have successfully logged out!" });
};  