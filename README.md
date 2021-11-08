You will need to have the `gcc` compiler installed.

You wil need to `chmod -R 555 test_input_data/` for the tests to pass (our code checks that chaced files are in mode 555 before readnig them to ensure that they are not corrupted.)

Please run the FULL test suit at least twice (i.e. don't use `-x` flag in pytest). The first time tests might fail because some things are getting installed by the test code (such as FastTree and XRATE), and the order of execution causes tests to fail. On the bright side, we install all required software for you.
