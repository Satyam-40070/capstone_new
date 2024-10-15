import User from '../models/userModel.js';
//import nodemailer from 'nodemailer';
import bcrypt from 'bcryptjs';
import jwt from 'jsonwebtoken';
import dotenv from 'dotenv';

dotenv.config();

const isValidEmail = (email) => {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return emailRegex.test(email);
};
/*
const sendVerificationEmail = async (email, res) => {
  try {
    const transporter = nodemailer.createTransport({
      service: "gmail",
      auth: {
        user: process.env.EMAIL_USER,
        pass: process.env.EMAIL_PASS
      },
    });

    const mailOptions = {
      from: process.env.EMAIL_USER,
      to: email,
      subject: "Registration Successful",
      html: `<p>Congratulations!! <br/>You have successfully registered to CodeVerse</p>`,
    };

    transporter.sendMail(mailOptions, (error, info) => {
      if (error) {
        console.error("Error sending email:", error);
        return res.status(500).json({ status: "Failed to send email" });
      }
      console.log("Email sent:", info.response);
      return res.status(200).json({ status: "Success" });
    });

  } catch (error) {
    console.error("Error in sending verification email:", error);
    return res.status(500).json({ status: "Failed to send verification email" });
  }
};*/

export const Register = async (req, res) => {
  try {
    const { fullname, username, email, password } = req.body;
    console.log(req.body);
    if (!(fullname && username && email && password)) {
      return res.status(400).json({ message: "Please fill all fields" });
    }

    if (!isValidEmail(email)) {
      return res.status(400).json({ message: "Invalid email format" });
    }

    const existingUser = await User.findOne({ email });
    if (existingUser) {
      return res.status(400).send("User already exists");
    }

    //sendVerificationEmail(email, res);

    const hashPassword = bcrypt.hashSync(password, 10);

    const user = await User.create({
      fullname,
      username,
      email,
      password: hashPassword,
    });

    const token = jwt.sign({ id: user._id, email, role: user.role }, process.env.SECRET_KEY, {
      expiresIn: '1d'
    });

    user.password = undefined;
    res.status(201).json({
      message: "You have successfully registered!",
      user,
      token
    });

  } catch (error) {
    console.error("Error in registration:", error);
    res.status(500).json({ message: "Internal server error" });
  }
};
